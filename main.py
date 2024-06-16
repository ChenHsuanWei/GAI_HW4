import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torchvision import datasets
from PIL import Image
import numpy as np
import math

# 定義一個簡單的卷積神經網絡作為DIP模型
class DIP(nn.Module):
    def __init__(self):
        super(DIP, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.conv4(x)
        return x

# 加入噪音
def add_noise(img, noise_level):
    noise = torch.randn_like(img) * noise_level
    noisy_img = img + noise
    return torch.clamp(noisy_img, 0., 1.)

# 生成有噪音的影像
def generate_noisy_images(target_img, noise_levels):
    noisy_images = []
    for noise_level in noise_levels:
        noisy_img = add_noise(target_img, noise_level)
        noisy_images.append(noisy_img)
    return noisy_images

# 計算PSNR
def psnr(img1, img2):
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    pixel_max = 1.0
    return 20 * math.log10(pixel_max / math.sqrt(mse))

# 訓練DIP模型
def train_DIP(dip_model, target_img, noisy_images, num_epochs=500, lr=0.001):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(dip_model.parameters(), lr=lr)
    
    psnr_history = []
    os.makedirs('results', exist_ok=True)
    for epoch in range(num_epochs):
        dip_model.train()
        optimizer.zero_grad()
        
        output = dip_model(target_img)
        
        # 計算損失
        loss = criterion(output, noisy_images[epoch % len(noisy_images)])
        loss.backward()
        optimizer.step()
        
        # 計算PSNR
        current_psnr = psnr(output, noisy_images[epoch % len(noisy_images)])
        psnr_history.append(current_psnr)

        # 每50個epoch保存一次結果
        if epoch % 20 == 0:
            print(f'Epoch [{epoch}/{num_epochs}], Loss: {loss.item():.4f}, PSNR: {current_psnr:.4f}')
            save_image(output, f'GAI_HW4/results/output_{epoch}.png')

        # 提前停止條件（例如PSNR不再顯著提升）
        if len(psnr_history) > 340 and all(psnr_history[-1] <= psnr for psnr in psnr_history[-10:]):#len(psnr_history) > 300 and
            print("Early stopping triggered")
            break

# 主函數
def main():
    # 加載並預處理目標影像
    #print("Current working directory:", os.getcwd())
    transform = transforms.Compose([transforms.ToTensor()])
    target_img = transform(Image.open(r'GAI_HW4\YehShuhua.png')).unsqueeze(0)
    
    # 定義噪音水平
    noise_levels = [0.1, 0.2, 0.3, 0.4]
    noisy_images = generate_noisy_images(target_img, noise_levels)
    
    # 初始化並訓練DIP模型
    dip_model = DIP()
    train_DIP(dip_model, target_img, noisy_images)
    
if __name__ == '__main__':
    main()