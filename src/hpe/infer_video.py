import copy

import cv2
import matplotlib.pyplot as plt
import numpy as np

import model
import util

from .body import Body
from .hand import Hand


def infer_video(video_path, body_weight, hand_weight):
    # build model and load weight
    body_estimation = Body(body_weight)
    hand_estimation = Hand(hand_weight)

    cap = cv2.VideoCapture(video_path)
    while True:
        ret, oriImg = cap.read()
        if not ret:
            break
        candidate, subset = body_estimation(oriImg)
        canvas = copy.deepcopy(oriImg)
        canvas = util.draw_bodypose(canvas, candidate, subset)

        # detect hand
        hands_list = util.handDetect(candidate, subset, oriImg)

        all_hand_peaks = []
        for x, y, w, is_left in hands_list:
            # 右手のみ対応
            if is_left:
                continue
            peaks = hand_estimation(oriImg[y:y + w, x:x + w, :])
            peaks[:, 0] = np.where(peaks[:, 0] == 0, peaks[:, 0], peaks[:, 0] + x)
            peaks[:, 1] = np.where(peaks[:, 1] == 0, peaks[:, 1], peaks[:, 1] + y)
            all_hand_peaks.append(peaks)

    canvas = util.draw_handpose(canvas, all_hand_peaks)
