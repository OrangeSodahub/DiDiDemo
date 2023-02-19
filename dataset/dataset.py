import os
import cv2
import numpy as np
from pathlib import Path
from paddle.io import Dataset
from paddle.vision.transforms import resize

class CRPDDataset(Dataset):
    def __init__(self, root_path: Path, split: str, str2int: dict):
        super().__init__()
        # For now only use CRPD dataset and has no imagesets
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

