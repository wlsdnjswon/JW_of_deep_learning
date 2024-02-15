from simple_unet_model import simple_unet_model   #unet model 가져오기
from keras.utils import normalize
import os
import cv2
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt

#데이터셋 위치
image_directory = 'C:/testAI/1/'
mask_directory = 'C:/testAI/2/'

#데이터셋 설정
SIZE = 512
image_dataset = []  
mask_dataset = []  

#이미지 읽어오기
images = os.listdir(image_directory)
#enumerate() 함수-> i ,image_name는 원소에 대한 숫자, images 안의 원소를 만들어줌
for i, image_name in enumerate(images):   
    if (image_name.split('.')[1] == 'png'):
        image = cv2.imread(image_directory+image_name, cv2.COLOR_BGR2RGB)
        image_dataset.append(np.array(image))

#마스크 읽어오기
masks = os.listdir(mask_directory)
for i, image_name in enumerate(masks):
    if (image_name.split('.')[1] == 'png'):
        image = cv2.imread(mask_directory+image_name, 0)
        image = Image.fromarray(image)
        image = image.resize((SIZE, SIZE))
        mask_dataset.append(np.array(image))

#normalize 함수를 사용해서 정규화
image_dataset = normalize(np.array(image_dataset), axis=(2,3))

#255로 나누어 0-1로 정규화
#image_dataset = np.array(image_dataset) /255.

#마스크 정규화 expand_dims함수를 통해 2차원 이미지를 3차원형태로 만들어줌
mask_dataset = np.expand_dims((np.array(mask_dataset)),3) /255.

#image_dataset과 mask_dataset은 (데이터셋개수, 가로, 세로, 채널)형식으로 이루어져 있음

#데이터 분할
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(image_dataset, mask_dataset, test_size = 0.10, random_state = 0)

#학습 데이터셋 이미지 보기
import random
import numpy as np
image_number = random.randint(0, len(X_train))
plt.figure(figsize=(12, 6))
plt.subplot(121)
plt.imshow(np.reshape(X_train[image_number], (SIZE, SIZE, 3)))
plt.subplot(122)
plt.imshow(np.reshape(y_train[image_number], (SIZE, SIZE)), cmap='gray')
plt.show()

###############################################################
#데이터 모양 가져와 모델 생성
IMG_HEIGHT = image_dataset.shape[1]
IMG_WIDTH  = image_dataset.shape[2]
IMG_CHANNELS = image_dataset.shape[3]

def get_model():
    return simple_unet_model(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)

model = get_model()


#사전훈련된 가중치로 학습할 경우 
#model.load_weights('mitochondria_gpu_tf1.4.hdf5')

history = model.fit(X_train, y_train, 
                    batch_size = 8, 
                    verbose=1, 
                    epochs=15, 
                    validation_data=(X_test, y_test), 
                    shuffle=False)

model.save('test.hdf5')

############################################################
#모델평가


	# 모델 정확도
_, acc = model.evaluate(X_test, y_test)
print("Accuracy = ", (acc * 100.0), "%")


#각 에포크에 대한 정확도와 손실
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'y', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

#acc = history.history['acc']
acc = history.history['accuracy']
#val_acc = history.history['val_acc']
val_acc = history.history['val_accuracy']

plt.plot(epochs, acc, 'y', label='Training acc')
plt.plot(epochs, val_acc, 'r', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

##################################
#IOU

y_pred=model.predict(X_train)
y_pred_thresholded = y_pred[1] > 0.5
y_test_thresholded = y_test[1]

intersection = np.logical_and(y_test_thresholded, y_pred_thresholded)
union = np.logical_or(y_test, y_pred_thresholded)
iou_score = np.sum(intersection) / np.sum(union)
print("IoU socre is: ", iou_score)

#######################################################################
#사전학습된 모델로 이미지 예측
model = get_model()
model.load_weights('test.hdf5')

#test 셋에 있는 이미지 예측
test_img_number = random.randint(0, len(X_test))
test_img = X_test[test_img_number]
ground_truth=y_test[test_img_number]
test_img_input = np.expand_dims(test_img, 0)
prediction = (model.predict(test_img_input)[0,:,:,0] > 0.2).astype(np.uint8)

#외부 이미지를 가져와 예측 
test_img_other = cv2.imread('C:/testAI/1/[0002]BtmBF0_12.jpg')
test_img_other_norm = normalize(np.array(test_img_other), axis=(2,3))
test_img_other_input = np.expand_dims(test_img_other_norm, 0)
prediction_other = (model.predict(test_img_other_input)[0,:,:,:] > 0.2).astype(np.uint8)

#해당 이미지를 출력
plt.figure(figsize=(16, 8))
plt.subplot(231)
plt.title('Testing Image')
plt.imshow(test_img, cv2.COLOR_BGR2RGB)
plt.subplot(232)
plt.title('Testing Label')
plt.imshow(ground_truth[:,:,0], cmap='gray')
plt.subplot(233)
plt.title('Prediction on test image')
plt.imshow(prediction, cmap='gray')
plt.subplot(234)
plt.title('External Image')
plt.imshow(test_img_other, cv2.COLOR_BGR2RGB)
plt.subplot(235)
plt.title('Prediction of external Image')
plt.imshow(prediction_other, cmap='gray')
plt.show()










