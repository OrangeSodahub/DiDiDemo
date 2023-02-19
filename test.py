import os
import cv2
import yaml
import argparse
import logging
from pathlib import Path

import paddlehub as hub
from paddleocr import PaddleOCR, draw_ocr
from paddle.io import DataLoader
from dataset.dataset import CRPDDataset, CCPDDataset

"""
Use CRPD double datasets.
"""
def run(config: dict, local: bool = False, save_results: bool = False, print_results: bool = False, use_ip_camera: bool = False):
    # prepare datasets
    data_path = config['DATASETS']['CCPD']
    dataset = CCPDDataset(Path(data_path), 'test', config['DATASETS']['str2int'])
    batch_size = config['MODEL']['test_cfg']['batch_size']
    dataLoader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=4)
    # TODO: support not save results
    save_dir = config['DATASETS']['save_results']
    os.makedirs(save_dir, exist_ok=True)
    logging.info(f"======> Loading {len(dataset)} samples")

    logging.info("======> Loading model ")
    ocr = hub.Module(name=config['MODEL']['name'])

    logging.info("======> Testing start ")
    for idx, (img, label) in enumerate(dataLoader):
        print(type(img))
        print(label)
    #     results = ocr.recognize_text(
    #                         images=images,
    #                         use_gpu=True,
    #                         output_dir=save_dir,
    #                         visualization=True,
    #                         box_thresh=0.5,
    #                         text_thresh=0.5)

    # if print_results:
    #     for result in results:
    #         data = result['data']
    #         save_path = result['save_path']
    #         for infomation in data:
    #             print('text: ', infomation['text'], '\nconfidence: ', infomation['confidence'],
    #                                 '\ntext_box_position: ', infomation['text_box_position'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Demo run arg parser')
    parser.add_argument("--config", help="path to config file", metavar="FILE", required=False, default= './configs/base.yaml')
    parser.add_argument("--save", help="whether to save the results", action='store_true', required=False)
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        try:
            config = yaml.load(f, Loader=yaml.FullLoader)
        except:
            raise FileNotFoundError

    logging.getLogger().setLevel(logging.INFO)

    run(config, args.save)
