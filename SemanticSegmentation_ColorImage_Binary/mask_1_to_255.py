# 주석 값이 0과 1인 경우, 1인 주석값을 255로 바꾸어 저장한다. 
import numpy as np
import os 
import cv2

path = '' # 폴더 경로
os.chdir(path) # 해당 폴더로 이동
files = os.listdir(path) # 해당 폴더에 있는 파일 이름을 리스트 형태로 받음


for file in files:
    mask = cv2.imread(file,cv2.IMREAD_GRAYSCALE )
    new_mask = np.where(mask == 1, 255, mask)
    cv2.imwrite(f"C:/testAI/0to255/{file}",new_mask)
