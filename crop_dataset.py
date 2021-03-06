import cv2
import glob
from skimage import io
import numpy as np
from skimage.transform import rescale, resize

cropping = False
ref_point = []
list_pic = []
name_file = []
TARGET_SIZE = (1280,960)

def shape_selection(event, x, y, flags, param):
  global ref_point, cropping
  if event == cv2.EVENT_LBUTTONDOWN:
    ref_point = [(x, y)]
    cropping = True
  elif event == cv2.EVENT_LBUTTONUP:
    ref_point.append((x, y))
    cropping = False
	
    cv2.rectangle(image, ref_point[0], ref_point[1], (0, 255, 0), 2)
    cv2.imshow("image", image)

name_folder = "2020_02_25-chicken_coconut_soup"
food = "food/"+name_folder

for file_name in glob.glob(food + "/*.jpg"):
	list_pic.append(file_name)
	name_file.append(file_name.split("\\")[1])

print(list_pic[0])

image = cv2.imread(list_pic[0], cv2.IMREAD_COLOR)
# image = cv2.resize(image,TARGET_SIZE)
clone = image.copy()
cv2.namedWindow("image", cv2.WINDOW_NORMAL)
cv2.setMouseCallback("image", shape_selection)

while True:
	cv2.imshow("image", image)
	# img = cv2.imread(img,cv2.IMREAD_COLOR)
	# img = cv2.resize(img,TARGET_SIZE)
	key = cv2.waitKey(1) & 0xFF
	
	if key == ord("r"):
		image = clone.copy()

	elif key == ord("q"):
		break

	elif key == ord("c"):
		for j in range(len(name_file)):
			image = cv2.imread(list_pic[j], cv2.IMREAD_COLOR)
			# image = cv2.resize(image,TARGET_SIZE)
			crop_img = image[ref_point[0][1]:ref_point[1][1], ref_point[0][0]:ref_point[1][0]]
			
			cv2.imwrite("food/"+name_folder+"/"+name_file[j],crop_img)	
			print(name_file[j]+ " is saved !")
		break

cv2.waitKey(0)
cv2.destroyAllWindows()



