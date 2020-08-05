# api.py
# -*- coding: utf-8 -*-

import cv2

face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_alt2.xml")
# cat_cascade = cv2.CascadeClassifier("haarcascade_frontalcatface_extended.xml")
# smile_cascade = cv2.CascadeClassifier("haarcascade_smile.xml")
profile_cascade = cv2.CascadeClassifier("haarcascade_profileface.xml")
ds_factor = 0.7

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
    # cat_rects = cat_cascade.detectMultiScale(gray, 1.3, 5)
    flipped = cv2.flip(gray, 1)
    profile_rects2 = profile_cascade.detectMultiScale(flipped, 1.3, 5)
    height, width = flipped.shape

    itr = 1
    for (x, y, w, h) in face_rects:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(image, f'Face {str(itr)}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        itr = itr + 1

    for (x, y, w, h) in profile_rects:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 255), 2)
        cv2.putText(image, f'Face {str(itr)}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
        itr = itr + 1

    for (x, y, w, h) in profile_rects2:
        new_x1 = width - x
        new_x2 = width - x - w
        cv2.rectangle(image, (new_x1, y), (new_x2, y + h), (0, 255, 255), 2)
        cv2.putText(image, f'Face {str(itr)}', (new_x2, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
        itr = itr + 1

    cv2.putText(image, f'Total faces: {str(itr - 1)}', (0, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

    '''
    itr = 1
    for (x, y, w, h) in cat_rects:
        cv2.rectangle(image, (x, y), (x + w, y + h), (246, 161, 255), 2)
        cv2.putText(image, f'Cat_{str(itr)}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (246, 161, 255), 1)
        itr = itr + 1
    '''

    cv2.imshow("Face recognition", image)

    key = cv2.waitKey(20)
    if key == 27: # exit on ESC
        break

vc.release()
cv2.destroyWindow("Face recognition")
