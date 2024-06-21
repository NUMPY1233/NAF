import numpy as np

def get_H(dNum, dSize):
    H = np.zeros(dNum)
    for i in range(0, dNum):
        if i == 0:
            H[i] = 1 / (4 * dSize**2)
        elif (i) % 2 == 0:
            H[i] = 0
        else:
            H[i] = -1 / ((i) * np.pi * dSize)**2
    return H
def reconstruct_image(proj):
    # 读取投影数据和图像数据


    pNum = 128
    pSize = 1
    dNum = 128
    dSize = pNum * pSize / dNum
    L = dNum * dSize
    views = 160
    da = np.pi / views
    # 初始化重建图像
    rec = np.zeros((pNum, pNum))
    iproj = np.zeros((views, dNum))
    # 定义图像像素点的坐标
    img_temp = np.linspace(-pNum/2 + pSize/2, pNum/2 - pSize/2, pNum, endpoint=True)
    [imgX, imgY] = np.meshgrid(img_temp, img_temp)

    # 生成采样点的坐标
    det_temp = np.linspace(-L/2 + dSize/2, L/2 - dSize/2, dNum, endpoint=True)
    for view in range(views):
        # 给投影值补零
        tmp_proj = list(proj[view, :])
        length = len(tmp_proj)
        w_t = dNum
        l = length - w_t
        tmp_proj = np.array(tmp_proj + [0] * (dNum - 1))
        length1 = len(tmp_proj)
        H1 = get_H(length, dSize)
        H2 = H1[1:]
        H2 = H2[::-1]
        f_RL = np.hstack((H1, H2))
        f_RL = np.fft.fft(f_RL)
        f_RL = np.real(f_RL)

        # 傅里叶变换
        fft_tmp_proj = np.fft.fft(tmp_proj)
        f_RL[w_t:w_t+2*l] = 0

        # 乘上一个滤波函数
        filter_result = fft_tmp_proj * f_RL
        ifft_filter_result = np.fft.ifft(filter_result) * dSize / 2
        iproj[view, :] = np.real(ifft_filter_result[0:dNum])

    numbda = 1

    # 针对每个像素反投影
    for view in range(views):
        phi = da * view + np.pi / 2  # 将角度增加90度
        rotx = np.cos(phi) * 0.7*imgX + np.sin(phi) * 0.7*imgY
        rec += numbda * np.interp(rotx, det_temp, iproj[view, :], 0, 0) * da
    rec[rec < 0] = 0
    rec_normalized = (rec - np.min(rec)) / (np.max(rec) - np.min(rec))
    return rec_normalized

# 示例调用

