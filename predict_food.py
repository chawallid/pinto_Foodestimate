from keras.models import Model, load_model
from keras.layers import Dense, Flatten, Input
from keras.layers import Conv2D, MaxPool2D
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.inception_v3 import preprocess_input
from keras.callbacks import ModelCheckpoint
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import cv2
import glob
import pandas as pd
import itertools
import collections
from collections import defaultdict



BATCH_SIZE = 5
# MAX_EPOCH = 2
IMAGE_SIZE = (512,512)
# TRAIN_IM = 160
# VALIDATE_IM = 15
def center_crop(x, center_crop_size, **kwargs):
    centerw, centerh = x.shape[0]//2, x.shape[1]//2
    halfw, halfh = center_crop_size[0]//2, center_crop_size[1]//2
    return x[centerw-halfw:centerw+halfw+1,centerh-halfh:centerh+halfh+1, :]

def predict_10_crop(img, ix, top_n=5, plot=False, preprocess=True):
    flipped_X = np.fliplr(img)
    crops = [
        img[:512,:512, :], # Upper Left
        img[:512, img.shape[1]-512:, :], # Upper Right
        img[img.shape[0]-512:, :512, :], # Lower Left
        img[img.shape[0]-512:, img.shape[1]-512:, :], # Lower Right
        center_crop(img, (512, 512)),
        
        flipped_X[:512,:512, :],
        flipped_X[:512, flipped_X.shape[1]-512:, :],
        flipped_X[flipped_X.shape[0]-512:, :512, :],
        flipped_X[flipped_X.shape[0]-512:, flipped_X.shape[1]-512:, :],
        center_crop(flipped_X, (512, 512))
    ]
    if preprocess:
        crops = [preprocess_input(x.astype('float32')) for x in crops]

    y_pred = model.predict(np.array(crops))
    preds = np.argmax(y_pred, axis=1)
    best_idx = collections.Counter(preds).most_common(1)[0][0]
    acc1 = y_pred[0][best_idx]
    # top_n_preds= np.argpartition(y_pred, -top_n)[:,-top_n:]
    # best_pre = 0
    # idx = 0
    # for i in range(len(y_pred)):
    #     # val = y_pred[i][preds[i]]
    #     if( best_pre < y_pred[i][preds[i]]):
    #         best_pre = y_pred[i][preds[i]]
    #         idx = preds[i]

    # print(best_pre,idx)
    # if()
    # best_pred = idx
    # acc = best_pre
    
    
    return preds , acc1

def predict_image(img):
    preds , acc = predict_10_crop(np.array(img), 0)
    best_pred = collections.Counter(preds).most_common(1)[0][0]
    print(classname[best_pred],"Accuracy: %.2f%%" % (acc*100))
    plt.suptitle(str(classname[best_pred])+" : Accuracy = %.2f%%" % (acc*100) , color='g')
    plt.imshow(pic)

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix'
                        #   ,cmap=plt.cm.Blues
                          ):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


datagen = ImageDataGenerator(rescale=1./255)
test_generator = datagen.flow_from_directory(
    'food/test',
    shuffle=True,
    target_size=IMAGE_SIZE,
    batch_size=15,
    color_mode='rgb',
    class_mode='categorical')

#Test Model
model = load_model('class_food.h5')
predict = model.predict_generator(test_generator,steps=len(test_generator))
# score = model.evaluate_generator(test_food,steps=len(test_food))

predicted_class_indices=np.argmax(predict,axis=1)
classname = (test_generator.class_indices)
classname = dict((v,k) for k,v in classname.items())
label_class = []
for i in range(len(classname)):
    label_class.append(classname[i])

# print(classname[0],len(classname))
# print("label_name : ", label_class)
# print(pd.DataFrame(score))
# predictions = [classname[k] for k in predicted_class_indices]

# y_true = test_generator.classes
# y_pred = predicted_class_indices

#confusion_matrix -----------------------------------------------------------------
# cnf_matrix = confusion_matrix(y_true, y_pred)

# plt.figure()
# fig = plt.gcf()
# fig.set_size_inches(32, 32)
# plot_confusion_matrix(cnf_matrix, classes=label_class,
#                       title='Confusion matrix',
#                       cmap=plt.cm.cool)
# plt.show()
#confusion_matrix -----------------------------------------------------------------


#accuracy_food
# corrects = collections.defaultdict(int)
# incorrects = collections.defaultdict(int)
# member_class = []
# for i in  range(len(classname)):
#     count_mem  = 0 
#     for j in y_true:
#         if(i == j):
#             count_mem = count_mem + 1
#     # print(count_mem)
#     member_class.append(count_mem)
# # print("member_class" , member_class)
# for (pred, actual) in zip(y_pred, y_true):

#     if pred == actual:
#         corrects[actual] += 1
#     else:
#         incorrects[actual] += 1
    
# class_accuracies = {}
# for idx in range(30):
#     class_accuracies[idx] = (corrects[idx] / member_class[idx]) 

#list accurracy-------------------------------------------------------------
# list_accuracy = []
# sorted_class_accuracies = sorted(class_accuracies.items(), key=lambda x: -x[1])
# for c in sorted_class_accuracies :
#     tmp = [c[0], c[1]]
#     list_accuracy.append(tmp)
# print(pd.DataFrame(list_accuracy))
# plt.hist(list(class_accuracies.values()), bins=20)
# plt.title('Accuracy by Class Food')


#list accurracy-------------------------------------------------------------
path  = 'predict/*.jpg'
for img in glob.glob(path):
    pic = cv2.imread(img , cv2.IMREAD_UNCHANGED)
    pic = cv2.resize(pic,(512,512))
    print("IMG : " , img)
    # plt.imshow(pic)
    predict_image(pic)
plt.show()