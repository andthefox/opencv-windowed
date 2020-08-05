# api.py
# -*- coding: utf-8 -*-

import cv2
import numpy as np

face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_alt2.xml")
profile_cascade = cv2.CascadeClassifier("haarcascade_profileface.xml")

ds_factor = 0.6

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

    image = cv2.resize(image, None, fx=ds_factor, fy=ds_factor, interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    face_rects = face_cascade.detectMultiScale(gray, 1.3, 5)
    profile_rects = profile_cascade.detectMultiScale(gray, 1.3, 5)

    flipped = cv2.flip(gray, 1)
    profile_rects2 = profile_cascade.detectMultiScale(flipped, 1.3, 5)
    height, width = flipped.shape
    for key in profile_rects2:
        key[0] = width - key[0]
        key[2] = -key[2]

    arr = np.empty((0, 4), int)
    for key in face_rects:
        arr = np.append(arr, face_rects, axis=0)
        break
    for key in profile_rects:
        arr = np.append(arr, profile_rects, axis=0)
        break
    for key in profile_rects2:
        arr = np.append(arr, profile_rects2, axis=0)
        break

    combined_list = arr.tolist()
    #print(combined_list)

    result = cv2.groupRectangles(combined_list, 1, 5)
    itr = 1
    if len(result[0]) != 0:
        for (x, y, w, h) in result[0]:
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(image, f'Face {str(itr)}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
            itr = itr + 1
        total = len(result[0])
    else:
        result = arr
        for (x, y, w, h) in result:
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(image, f'Face {str(itr)}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
            itr = itr + 1
        total = len(result)
    # print (result)

    cv2.putText(image, f'Total faces: {str(total)}', (0, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
    cv2.imshow("Face recognition", image)

    key = cv2.waitKey(20)
    if key == 27:  # exit on ESC
        break

vc.release()
cv2.destroyWindow("Face recognition")
