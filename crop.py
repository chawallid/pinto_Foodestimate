import cv2
import glob
from skimage import io
import numpy as np
from skimage.transform import rescale, resize

 
# initialize the list of reference points and boolean indicating
# whether cropping is being performed or not
path_input = "original_pic/2019-10-04 Rice Porridge/*.jpg"
path_output = "crop/2019-10-04 Rice Porridge/"
refPt = []
cropping = False
TARGET_SIZE = (960,720)


def click_and_crop(event, x, y, flags, param):
	# grab references to the global variables
	global refPt, cropping
 
	# if the left mouse button was clicked, record the starting
	# (x, y) coordinates and indicate that cropping is being
	# performed
	if event == cv2.EVENT_LBUTTONDOWN:
		refPt = [(x, y)]
		cropping = True
 
	# check to see if the left mouse button was released
	elif event == cv2.EVENT_LBUTTONUP:
		# record the ending (x, y) coordinates and indicate that
		# the cropping operation is finished
		refPt.append((x, y))
		cropping = False
 
		# draw a rectangle around the region of interest
		cv2.rectangle(image, refPt[0], refPt[1], (0, 255, 0), 2)
		cv2.imshow("image", image)
        

 
# load the image, clone it, and setup the mouse callback function
cv_img = []
name_pic = []
print("Load image :")
for img in glob.glob(path_input):
	n = cv2.imread(img)
	txt = img.split("/")
	txt = txt[1].split("\\")
	print(txt[1]," is loaded.")
	name_pic.append(txt[1])
	cv_img.append(n)
			
print("open : ",len(cv_img))

image = cv2.resize(cv_img[0],TARGET_SIZE)
clone = image.copy()
cv2.namedWindow("image")
cv2.setMouseCallback("image", click_and_crop)

# keep looping until the 'q' key is pressed
while True:
	# display the image and wait for a keypress
    #  print(refPt)
	cv2.imshow("image", image)
	key = cv2.waitKey(1) & 0xFF
	if key == ord("r"):
			image = clone.copy()
 
	elif key == ord("c"):
			#  global refPt
			# roi = image[refPt[0][1]:refPt[1][1], refPt[0][0]:refPt[1][0]]
			for l in range(len(cv_img)):
				write_name = path_output+name_pic[l]
				im = cv2.resize(cv_img[l],TARGET_SIZE)
				crop = im[refPt[0][1]:refPt[1][1], refPt[0][0]:refPt[1][0]]
				cv2.imwrite(write_name,crop)		
				print(write_name,"is saved!!!")
			cv2.waitKey(0)
			break

cv2.destroyAllWindows()

