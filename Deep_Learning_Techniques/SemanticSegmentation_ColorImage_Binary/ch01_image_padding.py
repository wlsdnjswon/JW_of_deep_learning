#이미지에 제로패딩을 넣기위한 코드
#2440x2040 사이즈의 이미지를 512로 나누기 위해 2560x2048로 제로패딩을 넣어 사용

import cv2
import os
import numpy as np

#위아래 4 픽셀 가로세로 60픽셀 마진
margin = ((4,4), (60,60), (0, 0))
margin_mask = ((4,4), (60,60))

#이미지 폴더 파일을 불러와 제로패딩을 넣고 생성
path = './train_original/'
re_path = './train_padding/'
#os.chdir(path)
files_image = os.listdir(path)
    
image_list = cv2.imread(path+file, cv2.COLOR_BGR2RGB)
for file in files_image:
    image_list = cv2.imread(path+file, cv2.COLOR_BGR2RGB)
    output = np.pad(image_list, margin, 'constant')
    cv2.imwrite(re_path+str(file.split('.')[0])+'.png', output)
   
#마스크 폴더 파일을 불러와 제로패딩을 넣고 생성
path = './mask_original/'
re_path = './mask_padding/'
#os.chdir(path)
files_image = os.listdir(path)
    
for file in files_image:
    image_list = cv2.imread(path+file, cv2.IMREAD_GRAYSCALE)
    output = np.pad(image_list, margin_mask, 'constant')
    cv2.imwrite(re_path+str(file.split('.')[0])+'.png', output)