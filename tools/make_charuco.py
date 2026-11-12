#!/usr/bin/env python3
"""Make the ChArUco calibration image to show full-screen on a laptop or tablet.

  python tools/make_charuco.py            -> charuco_8x5_screen.png
After showing it full-screen, measure one black square with a ruler (mm) and put that in project.json -> board.square_mm.
"""
import cv2, numpy as np
d = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_100)
board = cv2.aruco.CharucoBoard((8, 5), 0.04, 0.03, d)
img = board.generateImage((1600, 1000), marginSize=40, borderBits=1)
canvas = np.full((1080, 1920), 255, np.uint8)
canvas[40:1040, 160:1760] = img
cv2.imwrite("charuco_8x5_screen.png", canvas)
print("wrote charuco_8x5_screen.png")
