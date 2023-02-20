import os
import cv2
import numpy as np
import logging
from pathlib import Path
from paddle.io import Dataset
from paddle.vision.transforms import resize

from .utils import ccpd2label

class CRPDDataset(Dataset):
    def __init__(self, root_path: Path, split: str, str2int: dict):
        super().__init__()
        # CRPD dataset and has no imagesets
        if not os.path.exists(root_path / 'Imagesets'):
            self.splits = [line.strip('.txt') for line in \
                                    os.listdir(root_path / Path(split) / 'labels')]

        self.root_path = root_path
        self.str2int = str2int
        self.split_path = root_path / split
        self.num_samples = len(self.splits)
        self.transform = resize

    def __getitem__(self, idx):
        id = self.splits[idx]
        img = cv2.imread(str(self.split_path / 'images' / f'{id}.jpg'))
        label = np.loadtxt(self.split_path / 'labels' / f'{id}.txt', dtype=str)
        img = self.transform(img, (1080, 1920))
        # TODO: fix CRPD
        for line in label:
            line = np.concatenate(
                [line[:9].astype(np.float64), np.array([self.str2int[0][line[9][0]]]), np.array([self.str2int[1][x] for x in line[9][1:]])])

        return img, label

    def __len__(self):
        return self.num_samples


class CCPDDataset(Dataset):
    def __init__(self, root_path: Path, split: str, str2int: dict):
        super().__init__()
        # CCPD dataste has no labels
        label_path = root_path / 'labels' / Path(split)
        if not os.path.exists(label_path):
            logging.info("=====> Generate label files for CCPD dataset.")
            os.makedirs(label_path, exist_ok=True)
            ccpd2label(root_path, label_path, Path(split))
        self.splits = [line.strip('.txt') for line in os.listdir(label_path)]

        self.root_path = root_path
        self.str2int = str2int
        self.label_path = label_path
        self.split_path = root_path / split
        self.num_samples = len(self.splits)
        self.transform = resize

    def __getitem__(self, idx):
        name = self.splits[idx]
        img = cv2.imread(str(self.split_path / f'{name}.jpg'))
        # TODO: read label directly from image name
        label = name.split('/')[-1].split('.')[0].split('-')[-3]
        return img, label

    def __len__(self):
        return self.num_samples

    def _format_results(self, results, print_results: bool = False):
        """Format raw outputs to certain type to eval with labelGT.
        """
        labelPred = []
        for result in results:
            sublabelPred = []
            data = result['data']
            # TODO: handle that when len(data) > 1 or == 0
            # make sure that only one element to output
            if len(data) > 0:
                information = data[0]
                for str in information['text']:
                    # Skip the characters that not exist
                    if self.str2int[0].get(str, None) is not None:
                        sublabelPred.append(self.str2int[0].get(str))
                    elif self.str2int[1].get(str, None) is not None:
                        sublabelPred.append(self.str2int[1].get(str))
                if print_results:
                    print('text: ', information['text'], '\nconfidence: ', information['confidence'],
                                        '\ntext_box_position: ', information['text_box_position'])
            labelPred.append(sublabelPred)

        return labelPred


