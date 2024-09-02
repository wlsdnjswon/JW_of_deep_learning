import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns

epochs = 10

train = pd.read_csv('./sign_mnist_train.csv')
test = pd.read_csv('./sign_mnist_test.csv')

# numpy 배열로 변경
train_data = np.array(train, dtype = 'float32')
test_data = np.array(test, dtype='float32')

# 클래스 이름 정의
class_names = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 
               'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y']

# Sanity check - 이미지 라벨 출력
i = random.randint(1, train.shape[0])
fig1, ax1 = plt.subplots(figsize=(2, 2))
plt.imshow(train_data[i, 1:].reshape((28, 28)))
print("Label for the image is: ", class_names[int(train_data[i, 0])])

# 데이터 분포 시각화
fig = plt.figure(figsize=(18, 18))
ax1 = fig.add_subplot(221)
train['label'].value_counts().plot(kind='bar', ax=ax1)
ax1.set_ylabel('Count')
ax1.set_title('Label')

# X와 y로 데이터 분리
X = train_data[:, 1:] / 255.0
y = train_data[:, 0]

# 학습 데이터와 검증 데이터로 분할
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# 원-핫 인코딩
y_train_cat = to_categorical(y_train, num_classes=25)
y_val_cat = to_categorical(y_val, num_classes=25)

y_test = test_data[:, 0]
y_test_cat = to_categorical(y_test, num_classes=25)

# Reshape for the neural network
X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
X_val = X_val.reshape(X_val.shape[0], 28, 28, 1)
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)

# Model 1
model1 = Sequential()

model1.add(Conv2D(32, (3, 3), input_shape=(28, 28, 1), activation='relu'))
model1.add(MaxPooling2D(pool_size=(2, 2)))
model1.add(Dropout(0.2))

model1.add(Conv2D(64, (3, 3), activation='relu'))
model1.add(MaxPooling2D(pool_size=(2, 2)))
model1.add(Dropout(0.2))

model1.add(Conv2D(128, (3, 3), activation='relu'))
model1.add(MaxPooling2D(pool_size=(2, 2)))
model1.add(Dropout(0.2))

model1.add(Flatten())

model1.add(Dense(128, activation='relu'))
model1.add(Dense(25, activation='softmax'))

model1.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
model1.summary()

# Training the CNN model1
history1 = model1.fit(X_train, y_train_cat, batch_size=128, epochs=epochs, verbose=1, validation_data=(X_val, y_val_cat))
model1.save('saved_models/model1.hdf5')

# Model 2
model2 = Sequential()

model2.add(Conv2D(32, (3, 3), input_shape=(28, 28, 1), activation='relu'))
model2.add(Conv2D(32, (3, 3), activation='relu'))
model2.add(MaxPooling2D(pool_size=(2, 2)))

model2.add(Conv2D(64, (3, 3), activation='relu'))
model2.add(Conv2D(64, (3, 3), activation='relu'))
model2.add(Conv2D(64, (3, 3), activation='relu'))
model2.add(MaxPooling2D(pool_size=(2, 2)))

model2.add(Conv2D(128, (3, 3), activation='relu'))
model2.add(Conv2D(25, (1, 1)))

model2.add(Flatten())

model2.add(Dense(25, activation='softmax'))

model2.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
model2.summary()

history2 = model2.fit(X_train, y_train_cat, batch_size=128, epochs=epochs, verbose=1, validation_data=(X_val, y_val_cat))
model2.save('saved_models/model2.hdf5')

# Model 3
model3 = Sequential()

model3.add(Conv2D(32, (3, 3), input_shape=(28, 28, 1), activation='relu'))
model3.add(MaxPooling2D(pool_size=(2, 2)))
model3.add(Dropout(0.2))

model3.add(Conv2D(64, (3, 3), activation='relu'))
model3.add(MaxPooling2D(pool_size=(2, 2)))
model3.add(Dropout(0.2))

model3.add(Flatten())

model3.add(Dense(25, activation='softmax'))

model3.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
model3.summary()

history3 = model3.fit(X_train, y_train_cat, batch_size=128, epochs=epochs, verbose=1, validation_data=(X_val, y_val_cat))
model3.save('saved_models/model3.hdf5')

# Model Ensemble
from keras.models import load_model

model1 = load_model('saved_models/model1.hdf5')
model2 = load_model('saved_models/model2.hdf5')
model3 = load_model('saved_models/model3.hdf5')

models = [model1, model2, model3]

preds = [model.predict(X_test) for model in models]
preds = np.array(preds)
summed = np.sum(preds, axis=0)

# argmax across classes
ensemble_prediction = np.argmax(summed, axis=1)

accuracy1 = accuracy_score(y_test, np.argmax(preds[0], axis=1))
accuracy2 = accuracy_score(y_test, np.argmax(preds[1], axis=1))
accuracy3 = accuracy_score(y_test, np.argmax(preds[2], axis=1))
ensemble_accuracy = accuracy_score(y_test, ensemble_prediction)

print('Accuracy Score for model1 = ', accuracy1)
print('Accuracy Score for model2 = ', accuracy2)
print('Accuracy Score for model3 = ', accuracy3)
print('Accuracy Score for average ensemble = ', ensemble_accuracy)

# Weighted average ensemble
weights = [0.4, 0.2, 0.4]

weighted_preds = np.tensordot(preds, weights, axes=((0), (0)))
weighted_ensemble_prediction = np.argmax(weighted_preds, axis=1)

weighted_accuracy = accuracy_score(y_test, weighted_ensemble_prediction)

print('Accuracy Score for weighted average ensemble = ', weighted_accuracy)ㅊ