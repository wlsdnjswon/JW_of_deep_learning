from simple_multi_unet_model import multi_unet_model
from keras.utils import normalize
import os
import glob
import cv2
import numpy as np
from matplotlib import pyplot as plt
from sklearn.preprocessing import LabelEncoder


#이미지 크기
SIZE_X = 256 
SIZE_Y = 256
n_classes=6 #클래스 개수



########################################################################
#image, mask 읽어오기
train_images = []
for directory_path in glob.glob("./aug_dataset/aug_image"):
    for img_path in glob.glob(os.path.join(directory_path, "*.jpg")):
        img = cv2.imread(img_path, 1)       
        #img = cv2.resize(img, (SIZE_Y, SIZE_X))
        train_images.append(img)
       
#넘파이 배열로 변환     
train_images = np.array(train_images)



train_masks = [] 
for directory_path in glob.glob("./aug_dataset/aug_mask"):
    for mask_path in glob.glob(os.path.join(directory_path, "*.png")):
        mask = cv2.imread(mask_path, 0)       
        #mask = cv2.resize(mask, (SIZE_Y, SIZE_X), interpolation = cv2.INTER_NEAREST)  #Otherwise ground truth changes due to interpolation
        train_masks.append(mask)
                
train_masks = np.array(train_masks)

np.unique(train_masks)

###############################################
#레이블 인코딩, 다중 차원 배열을 1차원으로 변경해서 인코딩을 진행
labelencoder = LabelEncoder()
n, h, w = train_masks.shape
train_masks_reshaped = train_masks.reshape(-1,)
train_masks_reshaped_encoded = labelencoder.fit_transform(train_masks_reshaped)
train_masks_encoded_original_shape = train_masks_reshaped_encoded.reshape(n, h, w)

np.unique(train_masks_encoded_original_shape)

#################################################
#train_images = np.expand_dims(train_images, axis=3)
#train_images = normalize(train_images, axis=1)
train_images = np.array(train_images) /255.
train_masks_input = np.expand_dims(train_masks_encoded_original_shape, axis=3)

#데이터 나누기
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(train_images, train_masks_input, test_size = 0.10, random_state = 0)

#데이터 1번 더 나누기
#X_train, X_do_not_use, y_train, y_do_not_use = train_test_split(X_train, y_train, test_size = 0.2, random_state = 0)

print("Class values in the dataset are ... ", np.unique(y_train))

from keras.utils import to_categorical
train_masks_cat = to_categorical(y_train, num_classes=n_classes)
y_train_cat = train_masks_cat.reshape((y_train.shape[0], y_train.shape[1], y_train.shape[2], n_classes))



test_masks_cat = to_categorical(y_test, num_classes=n_classes)
y_test_cat = test_masks_cat.reshape((y_test.shape[0], y_test.shape[1], y_test.shape[2], n_classes))

np.unique(train_masks_reshaped_encoded)

###############################################################
#클래스 가중치 변환 (클래스 불균형 완화)
from sklearn.utils import class_weight
class_weights = class_weight.compute_class_weight(class_weight = "balanced", 
                                                  classes = np.unique(train_masks_reshaped_encoded), 
                                                  y = train_masks_reshaped_encoded)

print("Class weights are...:", class_weights)
class_weights = {i:w for i,w in enumerate(class_weights)}
###############################################################
# 클래스 가중치 적용 loss 함수

from keras import backend as K
from tensorflow import keras

def weightedLoss(originalLossFunc, weightsList):

    def lossFunc(true, pred):

        axis = -1 #if channels last 
        #axis=  1 #if channels first


        #가장 큰 값을 가지는 요소의 인덱스를 반환  
        classSelectors = K.argmax(true, axis=axis)
            #if your loss is sparse, use only true as classSelectors

        #클래스 인덱스가 가중치 인덱스와 같으면 true(1)을 사용   
        #classSelectors = [K.equal(i, classSelectors) for i in range(len(weightsList))]
        one64 = np.ones(1, dtype=np.int64)
        classSelectors = [K.equal(one64[0]*i, classSelectors) for i in range(len(weightsList))]
        classSelectors = [K.cast(x, K.floatx()) for x in classSelectors]

        #각각의 클래스마다 가중치를 곱함
        weights = [sel * w for sel,w in zip(classSelectors, weightsList)]


        #예측의 각 요소에 대해 각각의 가중치가 있는 텐서를 결과로 줌
        weightMultiplier = weights[0]
        for i in range(1, len(weights)):
            weightMultiplier = weightMultiplier + weights[i]


        loss = originalLossFunc(true,pred) 
        loss = loss * weightMultiplier

        return loss
    return lossFunc



###############################################################
#모델 학습

IMG_HEIGHT = X_train.shape[1]
IMG_WIDTH  = X_train.shape[2]
IMG_CHANNELS = X_train.shape[3]

def get_model():
    return multi_unet_model(n_classes=n_classes, IMG_HEIGHT=IMG_HEIGHT, IMG_WIDTH=IMG_WIDTH, IMG_CHANNELS=IMG_CHANNELS)

model = get_model()
#model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.compile(optimizer='adam', loss= weightedLoss(keras.losses.categorical_crossentropy, class_weights), metrics=['accuracy'] )
model.summary()

history = model.fit(X_train, y_train_cat, 
                    batch_size = 16, 
                    verbose=1, 
                    epochs=30, 
                    validation_data=(X_test, y_test_cat), 
                    class_weight=class_weights,
                    shuffle=False)
                    


model.save('multi_test.hdf5')
model.save('my_model.keras')

############################################################
#모델 평가
_, acc = model.evaluate(X_test, y_test_cat)
print("Accuracy is = ", (acc * 100.0), "%")


###
#정확도와 손실 함수 시각화
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

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

plt.plot(epochs, acc, 'y', label='Training Accuracy')
plt.plot(epochs, val_acc, 'r', label='Validation Accuracy')
plt.title('Training and validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()


##################################
#모델 IOU 계산하기
#model = get_model()
model.load_weights('sandstone_50_epochs_catXentropy_acc.hdf5')  
from tensorflow.keras.models import load_model
model = load_model('my_model.keras')


y_pred=model.predict(X_test)
y_pred_argmax=np.argmax(y_pred, axis=3)

#keras 라이브러리 사용
from keras.metrics import MeanIoU
n_classes = 6
IOU_keras = MeanIoU(num_classes=n_classes)  
IOU_keras.update_state(y_test[:,:,:,0], y_pred_argmax)
print("Mean IoU =", IOU_keras.result().numpy())


#각 클래스 IOU
values = np.array(IOU_keras.get_weights()).reshape(n_classes, n_classes)
row = np.sum(values, axis=0)
column = np.sum(values, axis=1)

class_IoU = []
for i in range(n_classes): 
    class_IoU.append(values[i,i]/(row[i] + column[i]))
    print("IoU for class{} is: ".format(i), class_IoU[i])


#######################################################################
#이미지 예측
import random
test_img_number = random.randint(0, len(X_test))
test_img = X_test[test_img_number]
ground_truth=y_test[test_img_number]
#test_img_norm=test_img[:,:,0][:,:,None]
test_img_input=np.expand_dims(test_img, 0)
prediction = (model.predict(test_img_input))
predicted_img=np.argmax(prediction, axis=3)[0,:,:]


plt.figure(figsize=(12, 8))
plt.subplot(231)
plt.title('Testing Image')
plt.imshow(test_img[:,:,:])
plt.subplot(232)
plt.title('Testing Label')
plt.imshow(ground_truth[:,:,0], cmap='jet')
plt.subplot(233)
plt.title('Prediction on test image')
plt.imshow(predicted_img, cmap='jet')
plt.show()





