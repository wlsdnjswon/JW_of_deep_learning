from simple_multi_unet_model import multi_unet_model
import os
import numpy as np
from patchify import patchify, unpatchify
import cv2
import matplotlib.pyplot as plt

#위아래 14 픽셀 가로세로 24픽셀 마진
margin = ((243, 0), (231, 0), (0, 0))

#이미지 폴더 파일을 불러와 제로패딩을 넣고 생성
path = './dataset/image/'
image_name = "image_part_1 (19).jpg"
    
image_list = cv2.imread(path+image_name, cv2.COLOR_BGR2RGB)
output = np.pad(image_list, margin, 'constant')
cv2.imwrite("./test/"+str(image_name.split('.')[0])+'.png', output)

#미리 학습된 모델 가져오기
IMG_HEIGHT = 256
IMG_WIDTH  = 256
IMG_CHANNELS = 3
n_classes = 7

def get_model():
    return multi_unet_model(n_classes=n_classes, IMG_HEIGHT=IMG_HEIGHT, IMG_WIDTH=IMG_WIDTH, IMG_CHANNELS=IMG_CHANNELS)

model = get_model()

from tensorflow.keras.models import load_model
model = load_model('my_model.keras')


# 예측하는 대형 이미지 가져오기
large_image = cv2.imread('./test/image_part_1 (19).png', cv2.COLOR_BGR2RGB)
large_mask = large_image[:,:,0]
patches = patchify(large_image, (256,256,3), step=256)

predicted_patches = []
for i in range(patches.shape[0]):
    for j in range(patches.shape[1]):
        print(i,j)
        
        single_patch = patches[i, j, 0, :, :, :]
        single_patch_norm = np.array(single_patch) /255.
        single_patch_input=np.expand_dims(single_patch_norm, 0)

        prediction = (model.predict(single_patch_input))
        predicted_img=np.argmax(prediction, axis=3)[0,:,:]
        predicted_patches.append(predicted_img)

predicted_patches = np.array(predicted_patches)
# 예측된 패치들을 (63, 512, 512, 1) 형태에서 (7, 9, 512, 512) 형태로 바꿈
predicted_patches_reshaped = np.reshape(predicted_patches, (patches.shape[0], patches.shape[1], 256, 256) )
# 예측된 패치를 원래 이미지 사이즈로 되돌리기
reconstructed_image = unpatchify(predicted_patches_reshaped, large_mask.shape)
plt.imshow(reconstructed_image)
plt.imsave('./test/segm.png', reconstructed_image)

cropped_img = reconstructed_image[243:, 231:]
plt.imshow(cropped_img)
plt.imsave('./test/cropped_segm.png', cropped_img)

plt.figure(figsize=(8, 8))
plt.subplot(221)
plt.title('Large Image')
plt.imshow(image_list)
plt.subplot(222)
plt.title('Prediction of large Image')
plt.imshow(cropped_img)
plt.show()



