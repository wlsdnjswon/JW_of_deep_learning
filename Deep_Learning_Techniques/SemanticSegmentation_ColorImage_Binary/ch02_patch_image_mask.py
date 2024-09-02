
import os
import numpy as np
from patchify import patchify, unpatchify
import cv2



# patchify_image

path = './train_padding/' # 폴더 경로
re_path = './train_patch/'
#os.chdir(path) # 해당 폴더로 이동
files_image = os.listdir(path) # 해당 폴더에 있는 파일 이름을 리스트 형태로 받음


for file in files_image:
    image_list = cv2.imread(path+file, cv2.COLOR_BGR2RGB)
    patches_img = patchify(image_list, (512,512, 3), step=512)
    for i in range(patches_img.shape[0]):
        for j in range(patches_img.shape[1]):
            single_patch_img = patches_img[i, j, 0, :, :, :]
            if not cv2.imwrite(re_path + str(file.split('.')[0]) + '_'+ str(i)+str(j)+'.png', single_patch_img):
                raise Exception("Could not write the image")


# patchify_mask

path = './mask_padding/'
re_path = './mask_patch/'
#os.chdir(path)
files_mask = os.listdir(path)


for file in files_mask:
    mask_list = cv2.imread(path+file, cv2.IMREAD_GRAYSCALE)
    patches_mask = patchify(mask_list, (512,512), step=512)
    for i in range(patches_mask.shape[0]):
        for j in range(patches_mask.shape[1]):
            single_patch_mask = patches_mask[i,j,:,:]
            if not cv2.imwrite(re_path + str(file.split('.')[0]) + '_'+ str(i)+str(j)+'.png', single_patch_mask):
                raise Exception("Could not write the image")




# Unpatchify_image

path = './train_padding/'
ph_path = './train_patch/'
re_path = './train_unpatch/'
files_image = os.listdir(path)

#기존의 이미지 및 패치 사이즈로 되돌리기 위한 제로 패딩을 만들어줌
img_shape = np.zeros([2048, 2560, 3])
unpatch = patchify(img_shape, (512,512,3), step=512)

for file in files_image:
    for i in range(unpatch.shape[0]):
        for j in range(unpatch.shape[1]):
            single_patch_img = cv2.imread(ph_path + str(file.split('.')[0]) + '_'+ str(i)+str(j)+'.png', cv2.COLOR_BGR2RGB) 
            if single_patch_img is None:
                raise Exception("Could not read the image") 
            #제로 패딩으로 만들어진 패치에 기존 패치를 복사     
            unpatch[i, j, 0, :, :, :] = single_patch_img.copy()
    
    reconstructed_image = unpatchify(unpatch, img_shape.shape)
    cv2.imwrite(re_path+ str(file.split('.')[0])+'.png', reconstructed_image)
    

# Unpatchify_mask

path = './mask_padding/'
ph_path = './mask_patch/'
re_path = './mask_unpatch/'
files_image = os.listdir(path)

mask_shape = np.zeros([2048, 2560])
unpatch = patchify(mask_shape, (512,512), step=512)

for file in files_image:
    for i in range(unpatch.shape[0]):
        for j in range(unpatch.shape[1]):
            single_patch_mask = cv2.imread(ph_path + str(file.split('.')[0]) + '_'+ str(i)+str(j)+'.png', cv2.IMREAD_GRAYSCALE) 
            if single_patch_mask is None:
                raise Exception("Could not read the image")    
            unpatch[i, j, :, :] = single_patch_mask.copy()
    
    reconstructed_mask = unpatchify(unpatch, mask_shape.shape)
    cv2.imwrite(re_path+ str(file.split('.')[0])+'.png', reconstructed_mask)



