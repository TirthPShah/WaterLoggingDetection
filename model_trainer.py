import os
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader
from torchvision import models
from tqdm import tqdm
import time

# --- CONFIGURATION ---
DATA_CSV = "./dataset/data.csv"
ROOT_DIR = "./dataset"
BATCH_SIZE = 8  # Adjust based on your GPU VRAM (Try 4 if 8 fails)
LR = 1e-4
EPOCHS = 25
DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using Device: {DEVICE}")

# --- 1. DATASET CLASS ---
class FloodDataset(Dataset):
    def __init__(self, dataframe, root_dir, transform=None):
        self.df = dataframe
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # Paths
        img_path = os.path.join(self.root_dir, row['image_path'])
        mask_path = os.path.join(self.root_dir, row['mask_path'])
        
        # Load Image
        image = cv2.imread(img_path)
        if image is None: # Safety for broken images
            print(f"Error loading {img_path}")
            image = np.zeros((512, 512, 3), dtype=np.uint8)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Load Mask
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            mask = np.zeros((512, 512), dtype=np.uint8)
        
        # Binarize (0 or 1.0)
        mask = (mask > 127).astype("float32")

        # Augmentations
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
        
        # Label
        label = torch.tensor(row['class_label'], dtype=torch.float32)

        return image, mask.unsqueeze(0), label

# --- 2. THE DUAL-BRANCH MODEL ---
class FloodAwareNet(nn.Module):
    def __init__(self):
        super(FloodAwareNet, self).__init__()
        
        # A. ENCODER (ResNet50)
        # We capture features at different scales for the U-Net
        base_model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.encoder0 = nn.Sequential(base_model.conv1, base_model.bn1, base_model.relu)
        self.pool0 = base_model.maxpool
        self.encoder1 = base_model.layer1 # 256 channels
        self.encoder2 = base_model.layer2 # 512 channels
        self.encoder3 = base_model.layer3 # 1024 channels
        self.encoder4 = base_model.layer4 # 2048 channels (Bottleneck)
        
        # B. SEGMENTATION DECODER (Simple U-Net style)
        self.upconv4 = nn.ConvTranspose2d(2048, 1024, kernel_size=2, stride=2)
        self.iconv4 = self.conv_block(1024 + 1024, 1024)
        
        self.upconv3 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.iconv3 = self.conv_block(512 + 512, 512)
        
        self.upconv2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.iconv2 = self.conv_block(256 + 256, 256)
        
        self.upconv1 = nn.ConvTranspose2d(256, 64, kernel_size=2, stride=2) # Note: encoder0 is 64
        self.iconv1 = self.conv_block(64 + 64, 64)
        
        self.final_up = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2) # to 256x256
        self.final_conv = nn.Conv2d(32, 1, kernel_size=1) 
        # Note: We will interpolate the final output to match input size exactly if needed
        
        # C. CLASSIFICATION HEAD (Gated)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 1)
        )

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # 1. Encoder Pass
        e0 = self.encoder0(x)
        p0 = self.pool0(e0)
        e1 = self.encoder1(p0)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        bottleneck = self.encoder4(e3)

        # 2. Decoder Pass
        d4 = self.upconv4(bottleneck)
        d4 = torch.cat((d4, e3), dim=1)
        d4 = self.iconv4(d4)
        
        d3 = self.upconv3(d4)
        d3 = torch.cat((d3, e2), dim=1)
        d3 = self.iconv3(d3)
        
        d2 = self.upconv2(d3)
        d2 = torch.cat((d2, e1), dim=1)
        d2 = self.iconv2(d2)
        
        d1 = self.upconv1(d2)
        if d1.shape != e0.shape:
            d1 = F.interpolate(d1, size=e0.shape[2:], mode='bilinear', align_corners=False)
        d1 = torch.cat((d1, e0), dim=1)
        d1 = self.iconv1(d1)
        
        out_mask_feat = self.final_up(d1)
        mask_logit = self.final_conv(out_mask_feat)
        mask_logit = F.interpolate(mask_logit, size=x.shape[2:], mode='bilinear', align_corners=False)
        pred_mask = torch.sigmoid(mask_logit)

        # 3. RESIDUAL GATING (The Fix)
        mask_small = F.interpolate(pred_mask, size=bottleneck.shape[2:], mode='area')
        
        # Formula: Features + (Features * Mask)
        # Result: Background = 1x Features, ROI = 2x Features.
        # This guarantees the classifier never sees "zeros".
        gated_features = bottleneck + (bottleneck * mask_small)
        
        pooled = self.global_pool(gated_features)
        pooled = pooled.flatten(1)
        class_logit = self.classifier(pooled)
        
        return class_logit, mask_logit, bottleneck

# --- 3. CUSTOM LOSS FUNCTION ---
class FloodLoss(nn.Module):
    def __init__(self, weight_cls=1.0, weight_seg=1.0, weight_bg=0.5):
        super(FloodLoss, self).__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.weight_cls = weight_cls
        self.weight_seg = weight_seg
        self.weight_bg = weight_bg

    def dice_loss(self, inputs, targets, smooth=1):
        inputs = torch.sigmoid(inputs)
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        intersection = (inputs * targets).sum()
        dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)
        return 1 - dice

    def background_suppression_loss(self, features, mask_gt):
        # features: [B, 2048, H_small, W_small]
        # mask_gt: [B, 1, H_orig, W_orig] -> needs downsampling
        
        # Downsample GT mask to feature size
        mask_small = F.interpolate(mask_gt, size=features.shape[2:], mode='nearest')
        
        # Invert Mask: We want to penalize features where Mask is 0 (Background)
        bg_mask = 1.0 - mask_small
        
        # Calculate Magnitude of features in background
        # L2 norm along channel dim
        feature_magnitude = torch.norm(features, p=2, dim=1, keepdim=True) # [B, 1, H, W]
        
        # We only want to minimize magnitude where bg_mask is 1
        penalty = (feature_magnitude * bg_mask).mean()
        return penalty

    def forward(self, class_logit, mask_logit, features, class_label, mask_gt):
        # 1. Classification Loss
        loss_c = self.bce(class_logit.view(-1), class_label)
        
        # 2. Segmentation Loss (Dice)
        loss_s = self.dice_loss(mask_logit, mask_gt)
        
        # 3. Background Suppression Loss
        # (Force the encoder features to be silent in non-annotated areas)
        loss_b = self.background_suppression_loss(features, mask_gt)
        
        total_loss = (self.weight_cls * loss_c) + \
                     (self.weight_seg * loss_s) + \
                     (self.weight_bg * loss_b)
        
        return total_loss, loss_c, loss_s, loss_b

# --- 4. TRAINING LOOP ---
def train():
    # Load Data
    df = pd.read_csv(DATA_CSV)
    train_df = df[df['split'] == 'train']
    val_df = df[df['split'] == 'val']
    
    transforms = A.Compose([
        A.Resize(320, 320), # Smaller size for speed/memory, increase to 512 later
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.2),
        A.Normalize(),
        ToTensorV2()
    ])
    
    train_loader = DataLoader(FloodDataset(train_df, ROOT_DIR, transforms), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(FloodDataset(val_df, ROOT_DIR, transforms), batch_size=BATCH_SIZE, shuffle=False)
    
    # Init Model
    model = FloodAwareNet().to(DEVICE)
    
    # --- FREEZE BACKBONE ---
    # We freeze the ResNet encoder to let the heads stabilize first
    print("Freezing Encoder weights...")
    for name, param in model.named_parameters():
        if "encoder" in name:
            param.requires_grad = False
    # -----------------------

    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LR)
    criterion = FloodLoss()
    
    print("Starting Training...")
    
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        for imgs, masks, labels in loop:
            imgs, masks, labels = imgs.to(DEVICE), masks.to(DEVICE), labels.to(DEVICE)
            
            optimizer.zero_grad()
            
            # Forward
            class_logit, mask_logit, features = model(imgs)
            
            # Loss
            loss, l_cls, l_seg, l_bg = criterion(class_logit, mask_logit, features, labels, masks)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            loop.set_postfix(loss=loss.item(), cls=l_cls.item(), seg=l_seg.item())
            
        # Validation
        model.eval()
        val_acc = 0
        total = 0
        with torch.no_grad():
            for imgs, masks, labels in val_loader:
                imgs, masks, labels = imgs.to(DEVICE), masks.to(DEVICE), labels.to(DEVICE)
                c_logit, m_logit, _ = model(imgs)
                
                preds = torch.sigmoid(c_logit) > 0.5
                val_acc += (preds.view(-1) == labels).sum().item()
                total += labels.size(0)
                
        print(f"Epoch {epoch+1} | Train Loss: {train_loss/len(train_loader):.4f} | Val Acc: {val_acc/total:.4f}")
        
        # Save Model
        torch.save(model.state_dict(), "flood_model.pth")

if __name__ == "__main__":
    train()