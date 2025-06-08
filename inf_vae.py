import torch
import torch.nn as nn
import torch.nn.functional as F
from models.cnn_vae import VAE
from depth2img import pointcloud_to_depth_image
import numpy as np
from torchvision import transforms
from PIL import Image

LATENT_DIM = 128

data_path = './data/defect8_2.pcd'
weights_path = './weights'

depth_img, depth_map = pointcloud_to_depth_image(
                pcd_path=data_path,
                distance_threshold=0.02,
                ransac_n=3,
                num_iterations=1000,
                resolution=0.0001,
                visualize=False,
                save_path=None
            )

depth_img = Image.fromarray(depth_img)
transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.Grayscale(num_output_channels=3),  # RGB로 변환
    transforms.ToTensor(),
])
input_tensor = transform(depth_img).unsqueeze(0)  # (1, 3, 512, 512)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = VAE(latent_dim=LATENT_DIM).to(device)
model.load_state_dict(torch.load(f'{weights_path}/vae_97percent.pth', map_location=device))
threshold = np.load(f'{weights_path}/threshold.npy')

model.eval()
with torch.no_grad():
    input_tensor = input_tensor.to(device)
    reconstructed, mu, logvar = model(input_tensor)
    errors = torch.sum((reconstructed - input_tensor) ** 2, dim=[1, 2, 3])
    recon_error = errors[0].item()  # 단일 이미지이므로 첫 번째 요소만 사용
    is_defective = recon_error > threshold
    print(f"Reconstruction Error: {recon_error:.4f}, 이상 유무: {is_defective}")


