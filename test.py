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


logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    filename='test.log', filemode='w', level=logging.INFO)
logger = logging.getLogger().setLevel(logging.INFO)
"""
Use CRPD double datasets.
"""
def eval(idx, labelGTs, labelPreds, count_list):
    """
    empty_count: number of [] in labelPreds
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
        logging.info(f"{idx}, GT: {labelGT}, Pred: {labelPred}, ratio: {float(count) / 7}\n")
    
    empty_count = 0
    for labelGT, labelPred in zip(labelGTs, labelPreds):
        if labelPred == []:
            empty_count += 1
            continue
        eval_single(labelGT, labelPred, count_list)
    
    return empty_count


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
    count = 0
    count_list = [0] * 8
    start = time()
    for idx, (imgs, labels) in enumerate(dataLoader()):
        # TODO: How to input tensor to ocr
        imgs = [img for img in imgs.numpy()]
        count += 1
        labelGT = [[int(ee) for ee in el.split('_')[:7]] for el in labels]
        results = ocr.recognize_text(
                            images=imgs,
                            use_gpu=False,
                            output_dir=save_dir,
                            visualization=True,
                            box_thresh=0.5,
                            text_thresh=0.5)

        labelPred = dataset._format_results(results, print_results)

        empty_count = eval(idx, labelGT, labelPred, count_list)
    
    # summary
    counter = 0
    for idx, count in enumerate(count_list):
        counter += idx * count
    print ('total %s precision %s precision_ %s fps %s' % \
         (len(dataset), float(counter)/(len(dataset)*7), float(counter)/((len(dataset)-empty_count)*7), len(dataset)/(time() - start)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Demo run arg parser')
    parser.add_argument("--config", help="path to config file", metavar="FILE", required=False, default= './configs/base.yaml')
    parser.add_argument("--print", help="whether to print the results in terminal", action='store_true', required=False)
    parser.add_argument("--save", help="whether to save the results", action='store_true', required=False)
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        try:
            config = yaml.load(f, Loader=yaml.FullLoader)
        except:
            raise FileNotFoundError

    # fh = logging.FileHandler('test.log').setLevel(logging.INFO)
    # ch = logging.StreamHandler().setLevel(logging.INFO)
    # formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    # fh.setFormatter(formatter)
    # ch.setFormatter(formatter)

    run(config, args.save, print_results=args.print)
