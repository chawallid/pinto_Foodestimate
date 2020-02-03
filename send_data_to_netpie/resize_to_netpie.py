import cv2
import numpy as np
def resize_img(path = "",position_img =""):
    TARGET_SIZE = (200,200)
    if(position_img == "center"):
        TARGET_SIZE = (400,200)
    im = cv2.imread(path)
    im_resized = cv2.resize(im,TARGET_SIZE)
    name_img = position_img+".jpg"
    cv2.imwrite(name_img,im_resized)

resize_img("IMG_20191003_162432.jpg","center")

  
 



