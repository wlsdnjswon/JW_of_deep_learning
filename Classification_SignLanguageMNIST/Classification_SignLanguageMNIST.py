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

# 데이터 로드
train = pd.read_csv('./sign_mnist_train.csv')
test = pd.read_csv('./sign_mnist_test.csv')

# numpy 배열로 변경
train_data = np.array(train, dtype = 'float32')
test_data = np.array(test, dtype='float32')

# 클래스 이름 정의
class_names = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 
               'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y']

#######################################################################################################
# 데이터 분포 시각화
#######################################################################################################
fig = plt.figure(figsize=(12, 8))  # 적당한 크기로 조정
ax1 = fig.add_subplot(111)

# 데이터 분포를 막대 그래프로 시각화
color_palette = sns.color_palette("viridis", n_colors=len(train['label'].unique()))  # 컬러 리스트 생성
value_counts = train['label'].value_counts().sort_index()  # 레이블 순서대로 정렬
# value_counts = train['label'].value_counts() # Count 순서대로 정렬

value_counts.plot(kind='bar', ax=ax1, color=color_palette)

# y축 및 x축 레이블 설정
ax1.set_ylabel('Count', fontsize=14)
ax1.set_xlabel('Label', fontsize=14)
ax1.set_title('Distribution of Labels in Training Data', fontsize=16)

# x축 레이블 회전 및 그리드 추가
ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45, ha='right', fontsize=12)
ax1.grid(True, axis='y', linestyle='--', alpha=0.7)

# 막대 위에 빈도 수 표시
for p in ax1.patches:
    ax1.annotate(f'{p.get_height()}', 
                 (p.get_x() + p.get_width() / 2., p.get_height()), 
                 ha='center', va='center', xytext=(0, 9), 
                 textcoords='offset points', fontsize=12)

plt.tight_layout()  # 레이아웃을 자동으로 조정하여 그래프가 겹치지 않도록 함
plt.show()
#######################################################################################################

# X와 y로 데이터 분리
X = train_data[:, 1:] / 255.0  # 정규화
y = train_data[:, 0]

# 학습 데이터와 검증 데이터로 분할
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# 원-핫 인코딩
y_train_cat = to_categorical(y_train, num_classes=25)
y_val_cat = to_categorical(y_val, num_classes=25)
y_test = test_data[:, 0]
y_test_cat = to_categorical(y_test, num_classes=25)

# 데이터 Reshape
X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
X_val = X_val.reshape(X_val.shape[0], 28, 28, 1)
X_test = (test_data[:, 1:] / 255.0).reshape(test_data.shape[0], 28, 28, 1)

# 모델 구성

model = Sequential()

model.add(Conv2D(32, (3, 3), input_shape=(28,28,1), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(25, activation='softmax'))

# 모델 컴파일
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

# 모델 학습
history = model.fit(X_train, y_train_cat, batch_size=128, epochs=10, verbose=1, validation_data=(X_val, y_val_cat))

# 학습 및 검증 손실 시각화
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)

plt.figure(figsize=(12, 6))
plt.plot(epochs, loss, 'o-', color='blue', label='Training Loss')  # 선 스타일과 색상 개선
plt.plot(epochs, val_loss, 'o-', color='orange', label='Validation Loss')
plt.title('Training and Validation Loss', fontsize=16)
plt.xlabel('Epochs', fontsize=14)
plt.ylabel('Loss', fontsize=14)
plt.xticks(epochs)  # x축에 에포크 번호를 명확히 표시
plt.legend(fontsize=12)
plt.grid(True)

# 마지막 에포크의 손실 값 표시
final_epoch = epochs[-1]
plt.text(final_epoch, loss[-1], f'{loss[-1]:.4f}', color='blue', ha='center', va='bottom', fontsize=12)
plt.text(final_epoch, val_loss[-1], f'{val_loss[-1]:.4f}', color='orange', ha='center', va='bottom', fontsize=12)

plt.show()

# 학습 및 검증 정확도 시각화
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

plt.figure(figsize=(12, 6))
plt.plot(epochs, acc, 'o-', color='green', label='Training Accuracy')  # 선 스타일과 색상 개선
plt.plot(epochs, val_acc, 'o-', color='red', label='Validation Accuracy')
plt.title('Training and Validation Accuracy', fontsize=16)
plt.xlabel('Epochs', fontsize=14)
plt.ylabel('Accuracy', fontsize=14)
plt.xticks(epochs)  # x축에 에포크 번호를 명확히 표시
plt.legend(fontsize=12)
plt.grid(True)

# 마지막 에포크의 정확도 값 표시
plt.text(final_epoch, acc[-1], f'{acc[-1]:.4f}', color='green', ha='center', va='bottom', fontsize=12)
plt.text(final_epoch, val_acc[-1], f'{val_acc[-1]:.4f}', color='red', ha='center', va='bottom', fontsize=12)

plt.show()

# 테스트 데이터 예측
predictions = model.predict(X_test)
predictions_classes = np.argmax(predictions, axis=1)

# 정확도 계산
accuracy = accuracy_score(y_test, predictions_classes)
print('Accuracy Score = ', accuracy)

# 임의 이미지 예측 결과 시각화
i = random.randint(1, len(predictions_classes))
plt.imshow(X_test[i,:,:,0], cmap='gray') 
print("Predicted Label: ", class_names[int(predictions_classes[i])])
print("True Label: ", class_names[int(y_test[i])])

# 혼동 행렬 시각화
cm = confusion_matrix(y_test, predictions_classes)
fig, ax = plt.subplots(figsize=(36,36))
sns.set(font_scale=1.6)
sns.heatmap(cm, annot=True, linewidths=.5, ax=ax)

# 잘못 분류된 비율 시각화
incorr_fraction = 1 - np.diag(cm) / np.sum(cm, axis=1)
fig, ax = plt.subplots(figsize=(12,12))
plt.bar(np.arange(24), incorr_fraction)
plt.xlabel('True Label')
plt.ylabel('Fraction of incorrect predictions')
plt.xticks(np.arange(25), class_names)
plt.show()















