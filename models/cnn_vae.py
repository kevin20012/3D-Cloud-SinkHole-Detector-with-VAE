import torch.nn as nn
import torch.nn.functional as F
import torch

class VAE(nn.Module):
    def __init__(self, latent_dim=128):
        super(VAE, self).__init__()
        # ---- 인코더: 512×512 → (채널256, 공간32×32) ----
        self.enc_conv1 = nn.Conv2d(3,   32,  kernel_size=4, stride=2, padding=1)  # 512→256
        self.enc_conv2 = nn.Conv2d(32,  64,  kernel_size=4, stride=2, padding=1)  # 256→128
        self.enc_conv3 = nn.Conv2d(64,  128, kernel_size=4, stride=2, padding=1)  # 128→64
        self.enc_conv4 = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1)  # 64→32

        # 어댑티브 풀링(선택) 또는 추가 Conv 레이어로 4×4까지 축소 가능
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))  # (채널256, 4×4)

        # ---- 잠재 변수 생성 ----
        self.fc_mu     = nn.Linear(256 * 4 * 4, latent_dim)
        self.fc_logvar = nn.Linear(256 * 4 * 4, latent_dim)

        # ---- 디코더: latent → (262144, 4×4) → 업샘플링(7단계) → (3, 512×512) ----
        self.fc_dec = nn.Linear(latent_dim, 256 * 4 * 4)

        # ConvTranspose 순서: 4→8→16→32→64→128→256→512 (총 7단계)
        self.dec_conv1 = nn.ConvTranspose2d(256, 256, kernel_size=4, stride=2, padding=1) # 4→8
        self.dec_conv2 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1) # 8→16
        self.dec_conv3 = nn.ConvTranspose2d(128, 64,  kernel_size=4, stride=2, padding=1) # 16→32
        self.dec_conv4 = nn.ConvTranspose2d(64,  32,  kernel_size=4, stride=2, padding=1) # 32→64
        self.dec_conv5 = nn.ConvTranspose2d(32,  16,  kernel_size=4, stride=2, padding=1) # 64→128
        self.dec_conv6 = nn.ConvTranspose2d(16,   8,  kernel_size=4, stride=2, padding=1) # 128→256
        self.dec_conv7 = nn.ConvTranspose2d(8,    3,  kernel_size=4, stride=2, padding=1) # 256→512

    def encode(self, x):
        x = F.relu(self.enc_conv1(x))  # 512→256
        x = F.relu(self.enc_conv2(x))  # 256→128
        x = F.relu(self.enc_conv3(x))  # 128→64
        x = F.relu(self.enc_conv4(x))  # 64→32
        x = self.adaptive_pool(x)      # 32→4
        x = x.view(x.size(0), -1)      # (배치, 256*4*4)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        x = self.fc_dec(z)
        x = x.view(-1, 256, 4, 4)          # (배치, 256, 4, 4)
        x = F.relu(self.dec_conv1(x))      # 4→8
        x = F.relu(self.dec_conv2(x))      # 8→16
        x = F.relu(self.dec_conv3(x))      # 16→32
        x = F.relu(self.dec_conv4(x))      # 32→64
        x = F.relu(self.dec_conv5(x))      # 64→128
        x = F.relu(self.dec_conv6(x))      # 128→256
        x = torch.sigmoid(self.dec_conv7(x))  # 256→512, 출력 범위 [0,1]
        return x

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)
        return recon_x, mu, logvar
