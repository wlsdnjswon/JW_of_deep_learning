from simple_unet_model import simple_unet_model 
from keras.utils import normalize
import os
import cv2
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
from patchify import patchify, unpatchify
from skimage import measure, color, io
import pandas as pd

def get_model():
    return simple_unet_model(512, 512, 3)

model = get_model()
# 학습된 가중치 가져오기
model.load_weights('test01.hdf5')

# 예측하는 대형 이미지 가져오기
large_image = cv2.imread('./train_padding/[0108]TopBF0.png', cv2.COLOR_BGR2RGB)
large_mask = large_image[:,:,0]
patches = patchify(large_image, (512,512,3), step=256)

# 이미지를 패치로 나누고 정규화하고 모델에 넣어 예측
predicted_patches = []
for i in range(patches.shape[0]):
    for j in range(patches.shape[1]):
        print(i,j)
        
        single_patch = patches[i, j, 0, :, :, :]
        single_patch_norm = np.array(single_patch) /255.
        single_patch_input=np.expand_dims(single_patch_norm, 0)

        # 0.5 확률 이상의 값에 대한 예측 0.4, 0.3으로도 예측할 수 있음
        single_patch_prediction = (model.predict(single_patch_input)[0,:,:,:] > 0.5).astype(np.uint8)
        predicted_patches.append(single_patch_prediction)

predicted_patches = np.array(predicted_patches)
# 예측된 패치들을 (63, 512, 512, 1) 형태에서 (7, 9, 512, 512) 형태로 바꿈
predicted_patches_reshaped = np.reshape(predicted_patches, (patches.shape[0], patches.shape[1], 512,512) )
# 예측된 패치를 원래 이미지 사이즈로 되돌리기
reconstructed_image = unpatchify(predicted_patches_reshaped, large_mask.shape)
plt.imshow(reconstructed_image, cmap='gray')
plt.imsave('./large_image/segm.png', reconstructed_image, cmap='gray')

plt.figure(figsize=(8, 8))
plt.subplot(221)
plt.title('Large Image')
plt.imshow(large_image, cmap='gray')
plt.subplot(222)
plt.title('Prediction of large Image')
plt.imshow(reconstructed_image, cmap='gray')
plt.show()

#########################################################
#Watershed를 사용해서 instance 픽셀로 바꾸기
#########################################################

img = cv2.imread('./large_image/segm.png')
img_gray = img[:,:,0]

ret1, thresh = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
plt.imshow(thresh, cmap='gray')

kernel = np.ones((3,3),np.uint8)
opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN,kernel, iterations = 2)
plt.imshow(opening, cmap='gray')

sure_bg = cv2.dilate(opening,kernel,iterations=10)
plt.imshow(sure_bg, cmap='gray')

dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,5)
plt.imshow(dist_transform, cmap='gray')

ret2, sure_fg = cv2.threshold(dist_transform, 0.05*dist_transform.max(),255,0)
plt.imshow(sure_fg, cmap='gray')

sure_fg = np.uint8(sure_fg)
unknown = cv2.subtract(sure_bg,sure_fg)
plt.imshow(unknown, cmap='gray')

ret3, markers = cv2.connectedComponents(sure_fg)
markers = markers+10
markers[unknown==255] = 0
plt.imshow(markers, cmap='gray')   

Wmarkers = cv2.watershed(img, markers)
plt.imshow(Wmarkers, cmap='gray') 


large_image[Wmarkers == -1] = [255,0,0]   
plt.imshow(large_image, cmap='gray')

props = measure.regionprops_table(Wmarkers, intensity_image=img_gray, 
                              properties=['label',
                                          'area', 'bbox', 'centroid', 'equivalent_diameter',
                                          'intensity_mean', 'solidity', ])

df = pd.DataFrame(props)
df = df[df.intensity_mean > 100]
print(df)

#large_image_원본 이미지 중심점 및 bbox 그림 그리기
for i in range(len(df)):
    (_, _, min_row, min_col, max_row, max_col, area_1, area_2, _, _, _) = df.iloc[i].astype(int)
    cv2.rectangle(large_image, (min_col, min_row, max_col-min_col, max_row-min_row), (255, 0, 0), 2)
    cv2.line(large_image, (area_2, area_1), (area_2, area_1), (0, 0, 255), 3)    

#img_예측 레이블 이미지 중심점 및 bbox 그림 그리기
for i in range(len(df)):
    (_, _, min_row, min_col, max_row, max_col, area_1, area_2, _, _, _) = df.iloc[i].astype(int)
    cv2.rectangle(img, (min_col, min_row, max_col-min_col, max_row-min_row), (255, 0, 0), 2)
    cv2.line(img, (area_2, area_1), (area_2, area_1), (0, 0, 255), 3)    


#img = cv2.imread('./large_image/segm.png')
#large_image = cv2.imread('./train_padding/[0108]TopBF0.png', 1)

plt.imshow(large_image)
plt.imsave('./large_image/Wmarkers_segm.png', Wmarkers)
cv2.imwrite('./large_image/predict_segm.png', img)
cv2.imwrite('./large_image/predict_large_segm.png', large_image)


# [시작 y좌표:끝 y좌표, 시작 x좌표:끝 x좌표]
# 파일을 area 영역만큼 crop(이미지를 자름)
cropped_img = large_image[4:2044, 60:2500] 
cv2.imwrite('./large_image/predict_large_cropped_segm.png', cropped_img)










