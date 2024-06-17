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

def add_noise(img, noise_level):
    noise = torch.randn_like(img) * noise_level
    noisy_img = img + noise
    return torch.clamp(noisy_img, 0., 1.)

def generate_noisy_images(target_img, noise_levels):
    noisy_images = []
    for noise_level in noise_levels:
        noisy_img = add_noise(target_img, noise_level)
        noisy_images.append(noisy_img)
    return noisy_images

def psnr(img1, img2):
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    pixel_max = 1.0
    return 20 * math.log10(pixel_max / math.sqrt(mse))

def train_DIP(dip_model, target_img, noisy_images, num_epochs=500, lr=0.001):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(dip_model.parameters(), lr=lr)
    
    psnr_history = []
    os.makedirs('results', exist_ok=True)
    for epoch in range(num_epochs):
        dip_model.train()
        optimizer.zero_grad()
        
        output = dip_model(target_img)
        
        loss = criterion(output, noisy_images[epoch % len(noisy_images)])
        loss.backward()
        optimizer.step()
        
        current_psnr = psnr(output, noisy_images[epoch % len(noisy_images)])
        psnr_history.append(current_psnr)

        # 每20個epoch保存一次結果
        if epoch % 15 == 0 and epoch>0:
            print(f'Epoch [{epoch}/{num_epochs}], Loss: {loss.item():.4f}, PSNR: {current_psnr:.4f}')
            save_image(output, f'GAI_HW4/results/output_{epoch}.png')

        # 停止條件
        if len(psnr_history) > 300 and all(psnr_history[-1] <= psnr for psnr in psnr_history[-10:]):#len(psnr_history) > 300 and
            print("Early stopping triggered")
            break
 
def main():
    #print("Current working directory:", os.getcwd())
    transform = transforms.Compose([transforms.ToTensor()])
    target_img = transform(Image.open(r'GAI_HW4\YehShuhua.png')).unsqueeze(0)
    
    noise_levels = [0.1, 0.2, 0.3, 0.4]
    noisy_images = generate_noisy_images(target_img, noise_levels)
    
    dip_model = DIP()
    train_DIP(dip_model, target_img, noisy_images)
    
if __name__ == '__main__':
    main()