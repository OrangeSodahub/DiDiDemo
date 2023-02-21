import os
import cv2
import yaml
import argparse
import logging
import numpy as np
from time import time
from pathlib import Path

import paddlehub as hub
from paddleocr import PaddleOCR, draw_ocr
from paddle.io import DataLoader
from dataset.dataset import CRPDDataset, CCPDDataset


"""
Use CRPD double datasets.
"""
def reset_log():
    fh = logging.FileHandler('test.log', 'a')
    sh = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    sh.setFormatter(formatter)
    log = logging.getLogger()
    for hdlr in log.handlers:
        log.removeHandler(hdlr)
    log.addHandler(fh)
    log.addHandler(sh)
    log.setLevel(logging.INFO)


def eval(idx, labelGTs, labelPreds, count_list):
    """
    count_list[8]: number of [] in labelPreds
    count_list[0]: none of matched between GTs and Preds
    """
    assert len(labelGTs) == len(labelPreds)
    def eval_single(labelGT, labelPred, count_list):
        # TODO: verify how to define the match
        count = 0
        labelPred_ = labelPred.copy()
        for g in labelGT:
            if g in labelPred_:
                count += 1
                index = labelPred_.index(g)
                labelPred_.pop(index)
        count_list[count] += 1
        logging.info(f"{idx}, GT: {labelGT}, Pred: {labelPred}, ratio: {float(count) / 7}")
    
    for labelGT, labelPred in zip(labelGTs, labelPreds):
        if labelPred == []:
            count_list[8] += 1
            continue
        eval_single(labelGT, labelPred, count_list)
    

def run(config: dict, local: bool = False, save_results: bool = False, print_results: bool = False, use_ip_camera: bool = False):
    # prepare datasets
    data_path = config['DATASETS']['CCPD']
    dataset = CCPDDataset(Path(data_path), 'test', config['DATASETS']['str2int'])
    batch_size = config['MODEL']['test_cfg']['batch_size']
    dataLoader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=4, places='cpu')
    # TODO: support not save results
    save_dir = config['DATASETS']['save_results']
    os.makedirs(save_dir, exist_ok=True)
    logging.info(f"======> Loading {len(dataset)} samples")

    logging.info("======> Loading model ")
    ocr = hub.Module(name=config['MODEL']['name'])

    logging.info("======> Testing start ")
    count_list = [0] * 9
    start = time()
    for idx, (imgs, labels) in enumerate(dataLoader()):
        # TODO: How to input tensor to ocr
        imgs = [img for img in imgs.numpy()]
        labelGT = [[int(ee) for ee in el.split('_')[:7]] for el in labels]
        results = ocr.recognize_text(
                            images=imgs,
                            use_gpu=False,
                            output_dir=save_dir,
                            visualization=True,
                            box_thresh=0.5,
                            text_thresh=0.5)

        labelPred = dataset._format_results(results, print_results)

        eval(idx, labelGT, labelPred, count_list)
    
    # summary
    counter = 0
    for idx, count in enumerate(count_list[:8]):
        counter += idx * count
    empty_count = count_list[8]
    logging.info('total %s precision %s precision_ %s\n fps %s count_list' % \
         (len(dataset), float(counter)/(len(dataset)*7), float(counter)/((len(dataset)-empty_count)*7),
         len(dataset)/(time() - start)), count_list)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Demo run arg parser')
    parser.add_argument("--config", help="path to config file", metavar="FILE", required=False, default= './configs/base.yaml')
    parser.add_argument("--print", help="whether to print the results in terminal", action='store_true', required=False)
    parser.add_argument("--save", help="whether to save the results", action='store_true', required=False)
    args = parser.parse_args()

    reset_log()
    with open(args.config, 'r') as f:
        try:
            config = yaml.load(f, Loader=yaml.FullLoader)
        except:
            raise FileNotFoundError

    run(config, args.save, print_results=args.print)
