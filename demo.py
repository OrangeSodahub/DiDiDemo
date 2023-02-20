import os
import cv2
import yaml
import argparse
import logging

import paddlehub as hub
from paddleocr import PaddleOCR, draw_ocr

def run(config: dict, local: bool = False, save_results: bool = False, print_results: bool = False, use_ip_camera: bool = False):
    # TODO: support not save results
    save_dir = config['DATASETS']['save_results']
    if local:
        imgs_folder_path = config['DATASETS']['test_folder']
        img_paths = os.listdir(imgs_folder_path)
        images = [cv2.imread(os.path.join(imgs_folder_path, img_path)) for img_path in img_paths]
        os.makedirs(save_dir, exist_ok=True)

    # Default to usb camera
    url = 0
    if use_ip_camera:
        url = config['IP_CAMERA']['url']
    cap = cv2.VideoCapture(url)

    logging.info("======> Loading model ")
    ocr = hub.Module(name=config['MODEL']['name'])

    logging.info("======> Running start ")
    while(cap.isOpened()):  
        ret, frame = cap.read()
        images = [frame]
        print(type(images))
        print(type(images[0]))
        print(images[0].shape)
        results = ocr.recognize_text(
                            images=images,
                            use_gpu=True,
                            output_dir=save_dir,
                            visualization=True,
                            box_thresh=0.5,
                            text_thresh=0.5)

        if print_results:
            for result in results:
                data = result['data']
                save_path = result['save_path']
                for infomation in data:
                    print('text: ', infomation['text'], '\nconfidence: ', infomation['confidence'],
                                        '\ntext_box_position: ', infomation['text_box_position'])
        cv2.imshow('frame',frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()  
    cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Demo run arg parser')
    parser.add_argument("--config", help="path to config file", metavar="FILE", required=False, default= './configs/base.yaml')
    parser.add_argument("--local", help="whether to run offline", action='store_true', required=False)
    parser.add_argument("--save", help="whether to save the results", action='store_true', required=False)
    parser.add_argument("--print", help="whether to print the results in terminal", action='store_true', required=False)
    parser.add_argument("--ip_camera", help="whether to use the ip camera", action='store_true', required=False)
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        try:
            config = yaml.load(f, Loader=yaml.FullLoader)
        except:
            raise FileNotFoundError
    
    run(config, args.local, args.save, args.print, args.ip_camera)
