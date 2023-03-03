import os
import cv2
import yaml
import argparse
import logging
import time
import numpy as np
from PIL import Image

from utils import init_args, draw_ocr_box_txt
from PaddleOCR.tools.infer.predict_system import TextSystem

def run(config: dict, args):
    # TODO: support not save results
    save_dir = config['DATASETS']['save_results']
    if args.local:
        imgs_folder_path = config['DATASETS']['test_folder']
        img_paths = os.listdir(imgs_folder_path)
        images = [cv2.imread(os.path.join(imgs_folder_path, img_path)) for img_path in img_paths]
        os.makedirs(save_dir, exist_ok=True)

    # Default to usb camera
    url = 0
    if args.ip_camera:
        url = config['IP_CAMERA']['url']
    cap = cv2.VideoCapture(url)

    logging.info("======> Loading model ")
    infer_sys = TextSystem(args)
    font_path = args.vis_font_path
    drop_score = args.drop_score
    # warm up 10 times
    if args.warmup:
        img = np.random.uniform(0, 255, [640, 640, 3]).astype(np.uint8)
        for _ in range(10):
            infer_sys(img)

    total_time = 0
    cpu_mem, gpu_mem, gpu_util = 0, 0, 0
    count = 0
    fps = 0
    start_time = time.time()
    logging.info("======> Running start ")
    while(cap.isOpened()):
        ret, image = cap.read()
        # image = cv2.imread('/home/zonlin/PaddlePaddle/DiDiDemo/data/CCPD2020/01d3-kcunqze6941645.jpg')
        dt_boxes, rec_res, time_dict = infer_sys(image)
        count += 1
        elaps = time.time() - start_time
        total_time += elaps
        fps = count / total_time

        if args.print:
            for text, score in rec_res:
                print("{}, {:.3f}".format(text, score))

        # visualization
        image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        boxes = dt_boxes
        txts = [rec_res[i][0] for i in range(len(rec_res))]
        scores = [rec_res[i][1] for i in range(len(rec_res))]

        draw_img = draw_ocr_box_txt(image, boxes, txts, scores,
                        drop_score=drop_score, font_path=font_path)

        cv2.imshow('results', np.array(draw_img))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    parser: argparse.ArgumentParser = init_args()
    parser.add_argument("--config", help="path to config file", metavar="FILE", required=False, default= './configs/base.yaml')
    parser.add_argument("--local", help="whether to run offline", action='store_true', required=False)
    parser.add_argument("--save", help="whether to save the results", action='store_true', required=False)
    parser.add_argument("--print", help="whether to print the results in terminal", action='store_true', required=False)
    parser.add_argument("--ip_camera", help="whether to use the ip camera", action='store_true', required=False)
    # End-to-end inference
    parser.add_argument("--det_model_dir", type=str, default='./models/det_ppocr_v3_finetune/infer')
    parser.add_argument("--rec_model_dir", type=str, default='./models/rec_ppocr_v3_finetune/infer')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        try:
            config = yaml.load(f, Loader=yaml.FullLoader)
        except:
            raise FileNotFoundError
    
    run(config, args)
