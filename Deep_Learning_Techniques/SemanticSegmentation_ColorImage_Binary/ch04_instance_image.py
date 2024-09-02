from simple_unet_model import simple_unet_model   
from keras.utils import normalize
import os
import cv2
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
from skimage import measure, color, io
import pandas as pd


IMG_HEIGHT = 512
IMG_WIDTH  = 512
IMG_CHANNELS = 3

def get_model():
    return simple_unet_model(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)

#모델과 가중치 로드
model = get_model()
model.load_weights('./test01.hdf5')

#테스트 이미지 로드
test_img = cv2.imread('./train_patch/[0108]TopBF0_33.png', cv2.COLOR_BGR2RGB)
test_img_norm = np.expand_dims(test_img, 0)
test_img_input = np.array(test_img_norm) /255.

#가중치를 가져온 모델을 가지고 예측 진행
segmented = (model.predict(test_img_input)[0,:,:,:] > 0.5).astype(np.uint8)

plt.figure(figsize=(8, 8))
plt.subplot(221)
plt.title('Testing Image')
plt.imshow(test_img, cmap='gray')
plt.subplot(222)
plt.title('Segmented Image')
plt.imshow(segmented, cmap='gray')
plt.show()


#차원을 줄이고 예측 이미지 저장
segmented = np.squeeze(segmented, axis=2)
plt.imsave('./instance_image/predict.png', segmented, cmap='gray')

########################################################


#이미지 로드
img = cv2.imread('./instance_image/predict.png')
img_gray = img[:,:,0]
plt.imshow(img_gray, cmap='gray')

#Threshold 함수를 사용하여 임계값을 적용하고 0아니면 255로 이미지 픽셀을 바꾸어 준다. ret1는 임계값
ret1, thresh = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

# Morphology 함수를 통해 커널 만큼의 열림 계산을 진행, 이미지의 노이즈 제거 효과
kernel = np.ones((3,3),np.uint8)
opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 2)
plt.imshow(opening, cmap='gray')


# dilate (팽창) 함수를 통해 커널만큼의 이미지 크기를 팽창, iterations는 반복 횟수로 조절 가능
sure_bg = cv2.dilate(opening,kernel, iterations=10)
plt.imshow(sure_bg, cmap='gray')

#distanceTransform(거리변환) 함수를 통해 픽셀값이 0인 부분에서 멀어지는 부분을 측정, 주석의 중앙으로 갈수록 값이 높아짐
dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,5)
plt.imshow(dist_transform, cmap='gray')

# 거리변환 함수를 통해 만들어진 이미지값의 최대값의 20%의 값이 임계값이 됌
# 임계값보다 높으면 255, 낮으면 0으로 설정한다., ret2는 임계값을 보여줌
ret2, sure_fg = cv2.threshold(dist_transform, 0.2*dist_transform.max(),255,0)
plt.imshow(sure_fg, cmap='gray')

# 예측된 이미지의 bkground에 foreground를 빼주면 각각의 주석을 255의 값으로 둘러 쌓기 가능
sure_fg = np.uint8(sure_fg)
unknown = cv2.subtract(sure_bg,sure_fg)
plt.imshow(unknown, cmap='gray')

# 이후 connectedComponents 함수를 통해 각각의 주석을 레이블링
# ret3는 객체의 개수, markers는 레이블맵으로 변환한 이미지를 출력
ret3, markers = cv2.connectedComponents(sure_fg)

# connectedComponentsWithStats 함수는 각 객체의 픽셀 개수 정보, 중심 위치 정보를 추가로 알려준다.
# retval, labels, stats, centroids = cv2.connectedComponentsWithStats(sure_fg)


# 배경이 0일 경우 watershed함수가 배경을 제대로 인식하지 못함
# 따라서 배경이 0이 아니라 10이 되도록 모든 레이블에 10을 추가
markers = markers+10

# markers를 unknown의 255픽셀 위치에 0으로 만들어줌
markers[unknown==255] = 0
plt.imshow(markers, cmap='gray')   

# 때문에 markers에는 확실하게 물체라고 확신하는 지역에 라벨값 (11, 12)을 놓고 
# 물체가 아니라고 확신하는 지역에 다른값 (10)으로 라벨을 붙임
# 최종적으로 아무것도 확신하지 못하는 지역에 0으로 라벨을 붙이며 watershed 함수는 경계 영역을 지정해준다.
Wmarkers = cv2.watershed(img, markers)
plt.imshow(Wmarkers, cmap='gray')


# 경계를 빨간색으로 변경
test_img[Wmarkers == -1] = [255,0,0]  
plt.imshow(test_img)

# 탐지된 영역 정보를 추출
# skimage measure module의 regionprops 함수는 각 객체에 대한 정보를 게산해서 보여줌
# label_레이블 번호, area_픽셀 수, bbox_각 박스 사이즈, centroid_중심좌표, 기타 등등
props = measure.regionprops_table(Wmarkers, intensity_image=img_gray, 
                              properties=['label',
                                          'area', 'bbox', 'centroid', 'equivalent_diameter',
                                          'intensity_mean', 'solidity', ])
#print(len(img_gray[img_gray == 255]))

df = pd.DataFrame(props)
# intensity_mean은 평균 강도를 나타냄, 객체로 집계될 수 있는 배경 또는 기타 영역 제거
df = df[df.intensity_mean > 100]
print(df.head())

# 각각의 객체 정보에 들어가기 위한 반복문.
for i in range(len(df)):
    (_, _, min_row, min_col, max_row, max_col, area_1, area_2, _, _, _) = df.iloc[i].astype(int)
    cv2.rectangle(test_img, (min_col, min_row, max_col-min_col, max_row-min_row), (255, 0, 0))
    cv2.line(test_img, (area_2, area_1), (area_2, area_1), (0, 0, 255), 3)    



test_img = cv2.imread('./train_patch/[0108]TopBF0_33.png', cv2.COLOR_BGR2RGB)
plt.imshow(test_img)






