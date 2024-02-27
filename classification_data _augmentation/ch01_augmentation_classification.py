import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import pandas as pd
import os
import cv2
from PIL import Image

plt.style.use('classic')

from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import BatchNormalization
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
from keras.datasets import cifar10
from keras.utils import normalize, to_categorical
from keras.layers import Dropout
from keras.optimizers import RMSprop, SGD

#from keras.optimizers.legacy import RMSprop, SGD

from sklearn.model_selection import train_test_split

########################################################################## 
#Input 데이터 외부에서 가져오기
(X_train, y_train), (X_test, y_test) = cifar10.load_data()
print("The size of training dataset X is: ", X_train.shape)
print("The size of testing dataset X is: ", X_test.shape)
print("The size of training dataset y is: ", y_train.shape)
print("The size of testing dataset y is: ", y_test.shape)

#데이터셋 크기 0.1로 줄이기
_, X, _, Y = train_test_split(X_train, y_train, test_size = 0.1, random_state = 0)
print("The size of the dataset X is: ", X.shape)
print("The size of the dataset Y is: ", Y.shape)


#이후 줄인 데이터셋으로 적은 양의 데이터셋 만들기
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.1, random_state = 0)
print("The size of training dataset is: ", X_train.shape)
print("The size of testing dataset is: ", X_test.shape)
print("The size of training dataset y is: ", y_train.shape)
print("The size of testing dataset y is: ", y_test.shape)


#train image 보기 
for i in range(9):
	plt.subplot(330 + 1 + i)
	plt.imshow(X_train[i])
plt.show()

#정규화
X_train = (X_train.astype('float32')) / 255.
X_test = (X_test.astype('float32')) / 255.

#마스크를 통한 데이터 카테고리 라벨링
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

################################################################
# 간단한 classfication 모델 만들기
drop=0.5

# 커널 초기값 
# 'zeros', 'random_uniform', 'he_uniform', 'glorot_uniform'등을 사용 가능
kernel_initializer =  'he_uniform'  

model1 = Sequential()
model1.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same', input_shape=(32, 32, 3)))
model1.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same'))
model1.add(MaxPooling2D((2, 2)))
model1.add(Dropout(drop))

model1.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same'))
model1.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same'))
model1.add(MaxPooling2D((2, 2)))
model1.add(Dropout(drop))

model1.add(Flatten())
model1.add(Dense(512, activation='relu', kernel_initializer=kernel_initializer))
model1.add(Dropout(drop))
model1.add(Dense(10, activation='softmax'))

opt1 = SGD(learning_rate=0.001, momentum=0.9)
opt2 = RMSprop(learning_rate=0.001)
model1.compile(optimizer=opt1, loss='categorical_crossentropy', metrics=['accuracy'])
model1.summary()
#################################################################
# Data augmentation

#너무 많은 회전 및 데이터 증강은 정확도를 떨어트림
#ImageDataGenerator는 텐서플로우 형식의 데이터 로더 형식으로 데이터 처리와 증강을 편하게 가능
train_datagen = ImageDataGenerator(rotation_range=15,  
    width_shift_range=0.3,  #try 0.1, 0.3
    height_shift_range=0.3, #try 0.1, 0.3
    zoom_range = 0.3, #try 0.1, 0.3
    vertical_flip=True,
    horizontal_flip = True,
    fill_mode="reflect")

#train_datagen.fit(X_train)

#train_datagen.flow를 통해 텐서플로우 데이터 로더 형식의 데이터 셋을 만듦
train_generator = train_datagen.flow(
    X_train,
    y_train,
    #데이터 로더 형식을 사용하면 미리 배치 사이즈를 정해놓아야 함
    batch_size = 32)

#train 데이터 셋에 접근해 1개의 정보를 가져옴 (배치사이즈 32이므로 32개의 32x32x3의 이미지와 라벨을 가져온다.)
x = train_generator.next()
print(x[0].shape)  #Images
print(x[1].shape)  #Labels
print((x[0].shape[0]))

#이미지 위쪽에 라벨 값을 출력
#0: airplane , 1: automobile, 2: bird, 3: cat, 4: deer, 5: dog, 6: frog, 7: horse, 8: ship, 9: truck
x = train_generator.next()
image = x[0][0]
title = np.argmax(x[1][0])
plt.figure(figsize=(1.5, 1.5))
plt.suptitle(title, fontsize=12)
plt.imshow(image)
plt.show()

batch_size = 32   
steps_per_epoch = len(X_train) // batch_size  

print("Total number of training images in the dataset = ", X_train.shape[0])
print("Steps per epoch = ", steps_per_epoch)
#fit_generator를 사용할 때 처리되는 샘플 수는 batch_size * step_per_epoch
print("Total data per epoch = ", steps_per_epoch*batch_size)


#모델 학습
history = model1.fit(
        train_generator,
        steps_per_epoch = steps_per_epoch,
        epochs = 25,
        validation_data = (X_test, y_test))

#모델 예측
_, acc = model1.evaluate(X_test, y_test)
print("Accuracy = ", (acc * 100.0), "%")

#증강을 했을때와 하지 않았을때의 차이
#실험에서 증강을 통한 정확도가 증강이 없는 경우에 비해 더 나쁘면 증강이 제대로 되지 않는다는 것을 의미
without_aug = {1000:36.4, 2000:45.2, 5000:51.7, 10000:58.4, 25000:69.4, 50000:77.3}
with_aug = {1000:44, 2000:48.4, 5000:54.7, 10000:60.8, 25000:70.7, 50000:78.4}
df = pd.DataFrame([without_aug, with_aug])
df = df.T
df.reset_index(inplace=True)
df.columns =['num_images', 'without_aug', 'with_aug']
print(df.head)
df.plot(x='num_images', y=['without_aug', 'with_aug'], kind='line')

#####################################################################
#각 에포크에서 훈련 및 검증 정확도 표시
history = history

#loss
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

#accuracy
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
plt.plot(epochs, acc, 'y', label='Training acc')
plt.plot(epochs, val_acc, 'r', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()











