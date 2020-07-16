#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 15 11:32:45 2020

@author: josh
"""


import cv2


vidcap = cv2.VideoCapture('seq_hotel.avi')
success, image = vidcap.read()

count = 0
while success:
    success, image = vidcap.read()
    cv2.imwrite("./frames/frame%d.jpg" % count, image)
    if cv2.waitKey(10)==27:
        break
    count+=1