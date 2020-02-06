import copy

import cv2
import numpy as np

import util

from .body import Body
from .hand import Hand


def infer_video(video_path, body_weight, hand_weight):
    """動画を推論して右手の各フレームにおける座標を返す

    Parameters
    ----------
    video_path : str
        input video path
    body_weight : str
        weight_path of body estimation model
    hand_weight : str
        weight_path of hand estimation model

    Returns
    -------
    np.ndarray(int)
        Array of right hand coordinates.
        right coordinates based on middle finger coordinates
    """
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

        right_hand_each_frame = []
        # not detected hand
        if not hands_list:
            right_hand_each_frame.append(np.nan)
        # get position of right hand
        for x, y, w, is_left in hands_list:
            # 右手のみ対応
            if is_left:
                continue
            peaks = hand_estimation(oriImg[y:y + w, x:x + w, :])
            peaks[:, 0] = np.where(peaks[:, 0] == 0, peaks[:, 0], peaks[:, 0] + x)
            peaks[:, 1] = np.where(peaks[:, 1] == 0, peaks[:, 1], peaks[:, 1] + y)

            # Set the middle finger coordinates to the right hand coordinates
            right_hand_each_frame.append(peaks[13, :])

    return np.asarray(right_hand_each_frame, dtype=int)
