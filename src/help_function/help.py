import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import radon
from scipy.io import loadmat
from .functions import reconstruct_image
# 假设有一个二维PET图像

def forward_project(pet_image,angle_num=160):
    image_center = None  # 图像中心，默认为图像中心
    angles = np.linspace(270., 90, num=angle_num, endpoint=False)  # 投影角度范围

# Radon 变换，生成 sinogram
    sinogram = radon(pet_image, theta=angles, circle=image_center)
    return sinogram



# 可视化 PET 图像和生成的 sinogram
if __name__== '__main__':
    pet_image = loadmat('/home/zyl/workspace/NAF/data/HD/image/train/img0.mat')['imgHD']
    pet_sino= loadmat('/home/zyl/workspace/NAF/data/HD/sinogram/train/sino0.mat')['sinoHD']
    pet_sino=(pet_sino-pet_sino.min())/(pet_sino.max()-pet_sino.min())
    sinogram = forward_project(pet_image)
    sinogram=(sinogram-sinogram.min())/(sinogram.max()-sinogram.min())
    sinogram= sinogram[27:155,:]
    print(pet_image.shape, sinogram.shape,pet_sino.shape)
    plt.figure(figsize=(15, 10))
    plt.subplot(1, 3, 1)
    plt.title('PET Image')
    plt.imshow(pet_image, cmap='gray')
    plt.subplot(1, 3, 2)
    plt.title('Sinogram')
    plt.xlabel('Projection angle (degrees)')
    plt.ylabel('Detector position')
    plt.imshow(sinogram, cmap='gray')
    plt.subplot(1, 3, 3)
    plt.title('pet_sino')
    plt.xlabel('Projection angle (degrees)')
    plt.ylabel('Detector position')
    plt.imshow(pet_sino, cmap='gray')
    plt.tight_layout()
    plt.show()


    img1=reconstruct_image(sinogram.T)
    img1=img1/img1.max()
    img2=reconstruct_image(pet_sino.T)
    img2=img2/img2.max()
    plt.figure(figsize=(15, 10))
    plt.subplot(1, 3, 1)
    plt.title('PET Image')
    plt.imshow(pet_image, cmap='gray')
    plt.subplot(1, 3, 2)
    plt.title('Reconstructed Image')
    plt.imshow(img1, cmap='gray')
    plt.subplot(1, 3, 3)
    plt.title('Reconstructed Image')
    plt.imshow(img2, cmap='gray')
    plt.tight_layout()
    plt.show()
