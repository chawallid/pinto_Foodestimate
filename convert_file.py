import whatimage
# import pyheif
import glob
import cv2
from PIL import Image
from skimage import io


path_input = "original_pic/2019-10-31 rice/*.HEIC"

for img in glob.glob(path_input):
    pic = cv2.imread(img)
    # size = pic.size
    text = img.split(".")
    text  = text[0].split("/")
    text = text[1].split("\\")
    write_name = text[1]+".jpg"
    cv2.imwrite(write_name,pic)
    print(text[1])
