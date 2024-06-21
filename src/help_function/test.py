import torch
import numpy as np
import matplotlib.pyplot as plt
from models import UNet
from functions import reconstruct_image
import scipy.io
# 验证集的FBP算法预处理图像
path = "y_test.npy"
y_test = np.load(path)
x_fbp = []
for i in range(y_test.shape[0]):
    reconstructed_image = reconstruct_image(path, i)
    x_fbp.append(reconstructed_image)
    if i % 10 == 0:
        print(f'FBP重建已进行到第{i+10}/{y_test.shape[0]}')
print(f'FBP重建已完成')

x_fbp = np.array(x_fbp)
np.save('x_test_fbp', x_fbp)

# 加载测试数据并转换为 PyTorch Tensor
test_data = torch.tensor(x_fbp, dtype=torch.float32).unsqueeze(1)

# 创建模型实例并加载已训练的权重
model = UNet()
model.load_state_dict(torch.load("unet_model.pth"))
model.eval()  # 设置为评估模式

# 使用模型进行预测
with torch.no_grad():
    outputs = model(test_data)

# 将输出保存
outputs_np = outputs.numpy()
x_test_rec=outputs.numpy().squeeze()
x_test_rec_transposed = np.transpose(x_test_rec, (1, 2, 0))
scipy.io.savemat('x_test_rec.mat', {'x_test_rec':x_test_rec_transposed})

# 可视化部分测试样本的输入和输出
num_samples = 5  # 指定要显示的样本数量
for i in range(num_samples):
    input_image = np.squeeze(test_data[i].numpy())
    output_image = np.squeeze(outputs_np[i])

    # 显示输入图像和输出图像
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(input_image, cmap='gray')
    plt.title("Input Image")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(output_image, cmap='gray')
    plt.title("Reconstructed Image")
    plt.axis("off")

    plt.show()
