import cv2
import glob
from skimage import io
import numpy as np
from skimage.transform import rescale, resize


Pt1 = [(52, 20), (405, 239)] #กลาง
Pt2 = [(232, 256), (407, 453)] #ซ้าย
Pt3 = [(58, 258), (221, 454)] #ขวา

TARGET_SIZE = (640,480)

def crop_food (img = ""):
	img = cv2.imread(img,cv2.IMREAD_COLOR)
	img = cv2.resize(img,TARGET_SIZE)
	clone = img.copy()

	while True:
			#draw line
			cv2.rectangle(img, Pt1[0], Pt1[1], (0, 255, 0), 2)
			cv2.rectangle(img, Pt2[0], Pt2[1], (0, 255, 0), 2)
			cv2.rectangle(img, Pt3[0], Pt3[1], (0, 255, 0), 2)
			cv2.imshow("Food tray :", img)
			key = cv2.waitKey(1) & 0xFF
			if key == ord("r"):

					img = clone.copy()

			elif key == ord("c"):

					crop1 = img[Pt1[0][1]:Pt1[1][1], Pt1[0][0]:Pt1[1][0]]
					crop2 = img[Pt2[0][1]:Pt2[1][1], Pt2[0][0]:Pt2[1][0]]
					crop3 = img[Pt3[0][1]:Pt3[1][1], Pt3[0][0]:Pt3[1][0]]
					
					crop1 =cv2.resize(crop1,(512,256))
					crop2 =cv2.resize(crop2,(256,256))
					crop3 =cv2.resize(crop3,(256,256))

					cv2.imwrite("img/center.jpg",crop1)	
					cv2.imwrite("img/left.jpg",crop2)	
					cv2.imwrite("img/right.jpg",crop3)	

					cv2.waitKey(0)
					break

	cv2.destroyAllWindows()


# crop_food("template.jpg")

