import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import torch
import torch.nn.functional as F
from models.cnn_vae import VAE
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report

def vae_loss_function(recon_x, x, mu, logvar):
    """
    VAE 손실 = 재구성 손실 + KL 발산
    recon_x: 디코더 출력 (재구성 이미지)
    x: 원본 입력 이미지
    mu: 잠재공간 평균
    logvar: 잠재공간 로그분산
    """
    recon_loss = F.mse_loss(recon_x, x, reduction='sum')
    
    # KL divergence
    kl_divergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    return recon_loss + kl_divergence, recon_loss, kl_divergence

class AnomalyImageDataset(Dataset):
    """
    디렉토리 내 JPG 파일을 확인하여, 파일명에 'normal'이 포함된 이미지는 정상(0), 
    포함되지 않은 이미지는 결함(1)으로 라벨링하는 데이터셋 클래스입니다.
    """
    def __init__(self, root_dir, transform=None):
        super().__init__()
        self.root_dir = root_dir
        self.transform = transform
        
        self.image_paths = [
            os.path.join(self.root_dir, fname)
            for fname in os.listdir(self.root_dir)
            if fname.lower().endswith('.jpg') or fname.lower().endswith('.jpeg') or fname.lower().endswith('.png')
        ]
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        
        # normal / anomaly labeling from filename
        if 'normal' in os.path.basename(img_path).lower():
            label = 0
        else:
            label = 1
        
        if self.transform:
            image = self.transform(image)
        return image, label


transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
])

full_dataset = AnomalyImageDataset(root_dir='data2img/', transform=transform)
train_image_indices = [i for i, (_, label) in enumerate(full_dataset) if label == 0] # extract only normal images for training
train_dataset = torch.utils.data.Subset(full_dataset, train_image_indices)

val_dataset = full_dataset

batch_size = 8
learning_rate = 1e-2
num_epochs = 100
latent_dim = 128

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = VAE(latent_dim=latent_dim).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

for epoch in range(1, num_epochs + 1):
    model.train()
    train_loss = 0.0
    for images, _ in train_loader:
        images = images.to(device)
        
        recon_images, mu, logvar = model(images)
        
        loss, recon_l, kl_l = vae_loss_function(recon_images, images, mu, logvar)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
    scheduler.step()
    # 에폭당 평균 손실 출력
    avg_train_loss = train_loss / len(train_loader.dataset)
    print(f"Epoch [{epoch}/{num_epochs}]  Train Loss: {avg_train_loss:.4f}")



model.eval()
all_recon_errors = []  
all_labels = []        

with torch.no_grad():
    for images, labels in val_loader:
        images = images.to(device)
        recon_images, mu, logvar = model(images)
        
        # 배치당 재구성 오차(MSE) 계산
        batch_errors = torch.sum((recon_images - images) ** 2, dim=[1, 2, 3])  
        all_recon_errors.extend(batch_errors.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

all_recon_errors = np.array(all_recon_errors)
all_labels = np.array(all_labels)

# reconstruction error threshold
normal_errors = all_recon_errors[all_labels == 0]
threshold = np.mean(normal_errors) + 3 * np.std(normal_errors)
np.save('./weights/threshold.npy', threshold)  # save threshold for future use
print(f"Reconstruction Error Threshold: {threshold:.4f}")

# thresholding to classify anomalies
pred_labels = (all_recon_errors > threshold).astype(int)  # 0: Normal, 1: Anomaly

accuracy = np.mean(pred_labels == all_labels)
print(f"Anomaly Detection Accuracy: {accuracy * 100:.2f}%")
torch.save(model.state_dict(), f'./weights/vae_{int(accuracy * 100)}percent.pth')  # save model

cm = confusion_matrix(all_labels, pred_labels)
print(f"Normal: {all_labels}")
print(f"Anomaly: {pred_labels}")
print("Confusion Matrix:")
print(cm)
print("Classification Report:")
print(classification_report(all_labels, pred_labels, target_names=["Normal", "Anomaly"]))

