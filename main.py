# api.py
# -*- coding: utf-8 -*-

import cv2
from mtcnn import MTCNN
import numpy as np

detector = MTCNN()

cv2.namedWindow("Face recognition", cv2.WINDOW_AUTOSIZE)
vc = cv2.VideoCapture(0)
vc.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
vc.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if vc.isOpened(): # try to get the first frame
    rval, frame = vc.read()
else:
    rval = False

while rval:
    rval, image = vc.read()

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    result = detector.detect_faces(image)

    bounding_box = result[0]['box']
    keypoints = result[0]['keypoints']

    cv2.rectangle(image,
                  (bounding_box[0], bounding_box[1]),
                  (bounding_box[0] + bounding_box[2], bounding_box[1] + bounding_box[3]),
                  (0, 155, 255),
                  2)

    cv2.circle(image, (keypoints['left_eye']), 2, (0, 155, 255), 2)
    cv2.circle(image, (keypoints['right_eye']), 2, (0, 155, 255), 2)
    cv2.circle(image, (keypoints['nose']), 2, (0, 155, 255), 2)
    cv2.circle(image, (keypoints['mouth_left']), 2, (0, 155, 255), 2)
    cv2.circle(image, (keypoints['mouth_right']), 2, (0, 155, 255), 2)

    # cv2.putText(image, f'Total faces: {str(total)}', (0, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
    cv2.imshow("Face recognition", image)

    key = cv2.waitKey(20)
    if key == 27:  # exit on ESC
        break

vc.release()
cv2.destroyWindow("Face recognition")
