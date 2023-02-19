import os
import cv2
import numpy as np
from tqdm import tqdm
from pathlib import Path


"""
Generate .txt label files for CCPD dataset.

"""

use_landmarks = False
use_xywh = False
use_ocr = True

def ccpd_landmarks(points_list, imgw, imgh):
    # four corners
    annotation = np.zeros((1,12))
    points_list_ = []
    for points in points_list.split("_"):
        points_ = list(map(int, points.split("&")))
        points_list_.append(points_)
    points_list_ = np.array(points_list_)

    # coords of box
    xmin = min(points_list_[:,0])
    xmax = max(points_list_[:,0])
    ymin = min(points_list_[:,1])
    ymax = max(points_list_[:,1])
    dw = 1./imgw
    dh = 1./imgh
    w, h = xmax - xmin, ymax - ymin

    # generate annotations
    annotation[0,0] = ((xmin + xmax) / 2. - 1) * dw #cx
    annotation[0,1] = ((ymin + ymax) / 2. - 1) * dh #cy
    annotation[0,2] = w * dw #w
    annotation[0,3] = h * dh #h
    annotation[0,4] = points_list_[2][0] / imgw
    annotation[0,5] = points_list_[2][1] / imgh
    annotation[0,6] = points_list_[3][0] / imgw
    annotation[0,7] = points_list_[3][1] / imgh
    annotation[0,8] = points_list_[0][0] / imgw
    annotation[0,9] = points_list_[0][1] / imgh
    annotation[0,10] = points_list_[1][0] / imgw
    annotation[0,11] = points_list_[1][1] / imgh

    return annotation


def ccpd_xywh(points_list, imgw, imgh):
    annotation = np.zeros((1, 4))
    points_list_ = []

    # four corners
    for points in points_list.split("_"):
        points_ = list(map(int, points.split("&")))
        points_list_.append(points_)
    points_list_ = np.array(points_list_)

    # coords of box
    xmin = min(points_list_[:,0])
    xmax = max(points_list_[:,0])
    ymin = min(points_list_[:,1])
    ymax = max(points_list_[:,1])
    dw = 1. / imgw
    dh = 1. / imgh
    w,h=xmax-xmin,ymax-ymin

    # generate annotations
    annotation[0, 0] = ((xmin + xmax) / 2. - 1) * dw  # cx
    annotation[0, 1] = ((ymin + ymax) / 2. - 1) * dh  # cy
    annotation[0, 2] = w * dw  # w
    annotation[0, 3] = h * dh  # h
    return annotation


def ccpd_ocr(plate):
    provinces = ["皖", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑", "苏", "浙", "京", "闽", "赣", "鲁", "豫", "鄂", "湘", "粤",
                 "桂", "琼", "川", "贵", "云", "藏", "陕", "甘", "青", "宁", "新", "警", "学", "O"]
    word_lists = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V',
                 'W','X', 'Y', 'Z', 'O', '1', '2', '3', '4', '5', '6', '7', '8', '9','0']

    label_list = plate.split("_")
    result = ""
    result += provinces[int(label_list[0])]
    result += word_lists[int(label_list[1])]
    result += word_lists[int(label_list[2])] + word_lists[int(label_list[3])] + word_lists[int(label_list[4])] +\
            word_lists[int(label_list[5])] + word_lists[int(label_list[6])]
    return result


def ccpd2label(root_path, txt_path, split):
    image_list = os.listdir(root_path / split)
    for imgpath in tqdm(image_list):
        img = cv2.imread(os.path.join(str(root_path / split), imgpath))
        h, w = img.shape[:2]

        imgname = Path(imgpath).name
        points_list, plate_list = imgname.split("-")[3], imgname.split("-")[4]
        if use_landmarks:
            annotation = ccpd_landmarks(points_list, w, h)
            txtname = imgname.replace(".jpg", ".txt")
            txt_path = os.path.join(txt_path, txtname)
            str_label = "0"
            with open(txt_path, "a+") as fw:
                for i in range(len(annotation[0])):
                    str_label = str_label + " " + str(annotation[0][i])
                fw.write(str_label)
        elif use_xywh:
            annotation = ccpd_xywh(points_list, w, h)
            txtname = imgname.replace(".jpg", ".txt")
            txt_path = os.path.join(txt_path, txtname)
            str_label = "0"
            with open(txt_path, "a+") as fw:
                for i in range(len(annotation[0])):
                    str_label = str_label + " " + str(annotation[0][i])
                fw.write(str_label)
        elif use_ocr:
            ocr_label = ccpd_ocr(plate_list)
            txtname = imgname.replace(".jpg", ".txt")
            save_path = os.path.join(txt_path, txtname)
            with open(save_path, "w") as fw:
                fw.write(ocr_label)
