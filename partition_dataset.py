import cv2
import glob
import shutil
import random as rd 

path_input = "crop/*"
subpath = "/*.jpg"
path_output = "food/"

image_size =(960,720)

test_num = 0
train_num = 0 
valid_num = 0 

Cross_validtion = 10
def partition_(img,test,train,valid,nameF):
    print("test:",test)
    print("train:",train)
    print("valid",valid)
    list_valid = []
    list_test = []
    list_all = []
    i = 0
    while i < valid:
        index_valid = rd.randrange(0,len(img))
        if((index_valid in list_valid) == False):
            i += 1
            list_valid.append(index_valid)
            list_all.append(index_valid)
    i = 0
    while i < test:
        index_test = rd.randrange(0,len(img))
        if((index_test in list_test) == False and (index_test in list_valid) == False):
            i += 1
            list_test.append(index_test)
            list_all.append(index_test)
    for i in list_valid :
        srt_name = img[i].split("\\")
        # print(path_output+"validation/"+nameF +"/"+ srt_name[2])
        shutil.move(img[i],path_output+"validation/"+nameF +"/"+ srt_name[2])
    for j in list_test: 
        srt_name = img[j].split("\\")
        # print(path_output+"test/"+nameF +"/"+ srt_name[2])
        shutil.move(img[j],path_output+"test/"+nameF +"/"+ srt_name[2])
    for i in range(len(img)):
        srt_name = img[i].split("\\")
        # print(path_output+"validation/"+nameF +"/"+ srt_name[1])
        if((i in list_test) == False and (i in list_valid) == False):
            shutil.move(img[i],path_output+"train/"+nameF +"/"+ srt_name[2])
    # # print(list_all)
    return 0 

def main(_c):
    for i in glob.glob(path_input):
        img_num = 0 
        cv_img = []
        img_target = i+subpath
        name_food = i.split("\\")
        print("name food:" , name_food[1])
        # print("food name :", name_food[1])
        for j in glob.glob(img_target):
            img_num += 1 
            cv_img.append(j)
        print("num :" + str(img_num))
        train_num = int(img_num *((100.0 - _c)/100.0))
        valid_num = int(img_num - train_num)
        test_num = valid_num
        partition_(cv_img,test_num,train_num,valid_num,name_food[1])
        print("------------------------------")

    return 0 
    
if __name__ == "__main__":
    print("cross_validation :", Cross_validtion)
    cross = Cross_validtion
    main(int(cross))
