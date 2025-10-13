"""
Flood Detection Classification Models

This module contains:
1. Custom CNN trained from scratch
2. Pretrained model wrappers (ResNet50, EfficientNet-B0, MobileNetV2)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from typing import Optional, Dict


class CustomFloodCNN(nn.Module):
    """
    Custom CNN architecture for flood detection (trained from scratch).
    
    Architecture:
    - 5 Convolutional blocks (Conv + BatchNorm + ReLU + MaxPool)
    - Progressive channel expansion: 3 → 32 → 64 → 128 → 256 → 512
    - Global Average Pooling
    - 2 Dense layers with dropout
    - Binary classification output (Sigmoid)
    
    Input: (batch, 3, 244, 244)
    Output: (batch, 1) - probability of flooding
    """
    
    def __init__(self, dropout_rate: float = 0.5):
        """
        Initialize Custom CNN.
        
        Args:
            dropout_rate: Dropout probability for regularization (default: 0.5)
        """
        super(CustomFloodCNN, self).__init__()
        
        # Convolutional Block 1: 3 → 32
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # 244 → 122
        
        # Convolutional Block 2: 32 → 64
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)  # 122 → 61
        
        # Convolutional Block 3: 64 → 128
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)  # 61 → 30
        
        # Convolutional Block 4: 128 → 256
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)  # 30 → 15
        
        # Convolutional Block 5: 256 → 512
        self.conv5 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(512)
        self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2)  # 15 → 7
        
        # Global Average Pooling
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Fully Connected Layers
        self.fc1 = nn.Linear(512, 256)
        self.dropout1 = nn.Dropout(dropout_rate)
        
        self.fc2 = nn.Linear(256, 128)
        self.dropout2 = nn.Dropout(dropout_rate)
        
        self.fc3 = nn.Linear(128, 1)  # Binary classification
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights using He initialization."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor (batch, 3, 244, 244)
            
        Returns:
            Output tensor (batch, 1) with sigmoid activation
        """
        # Conv Block 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool1(x)
        
        # Conv Block 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool2(x)
        
        # Conv Block 3
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.pool3(x)
        
        # Conv Block 4
        x = self.conv4(x)
        x = self.bn4(x)
        x = F.relu(x)
        x = self.pool4(x)
        
        # Conv Block 5
        x = self.conv5(x)
        x = self.bn5(x)
        x = F.relu(x)
        x = self.pool5(x)
        
        # Global Average Pooling
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)  # Flatten
        
        # Fully Connected Layers
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout1(x)
        
        x = self.fc2(x)
        x = F.relu(x)
        x = self.dropout2(x)
        
        x = self.fc3(x)
        x = torch.sigmoid(x)  # Binary classification
        
        return x
    
    def get_num_parameters(self) -> int:
        """Get total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class PretrainedFloodClassifier(nn.Module):
    """
    Pretrained model wrapper for flood detection.
    
    Supports:
    - ResNet50
    - EfficientNet-B0
    - MobileNetV2
    
    Strategy:
    - Load pretrained ImageNet weights
    - Freeze early layers (optional)
    - Replace final classification head
    - Fine-tune on flood detection task
    """
    
    def __init__(
        self,
        model_name: str = 'resnet50',
        pretrained: bool = True,
        freeze_backbone: bool = False,
        num_unfreeze_layers: int = 10
    ):
        """
        Initialize pretrained model.
        
        Args:
            model_name: One of ['resnet50', 'efficientnet_b0', 'mobilenet_v2']
            pretrained: Load ImageNet pretrained weights
            freeze_backbone: Freeze backbone layers
            num_unfreeze_layers: Number of layers to unfreeze from the end
        """
        super(PretrainedFloodClassifier, self).__init__()
        
        self.model_name = model_name
        
        # Load base model
        if model_name == 'resnet50':
            self.base_model = models.resnet50(pretrained=pretrained)
            num_features = self.base_model.fc.in_features
            self.base_model.fc = nn.Identity()  # Remove original FC
            
        elif model_name == 'efficientnet_b0':
            self.base_model = models.efficientnet_b0(pretrained=pretrained)
            num_features = self.base_model.classifier[1].in_features
            self.base_model.classifier = nn.Identity()  # Remove original classifier
            
        elif model_name == 'mobilenet_v2':
            self.base_model = models.mobilenet_v2(pretrained=pretrained)
            num_features = self.base_model.classifier[1].in_features
            self.base_model.classifier = nn.Identity()  # Remove original classifier
            
        else:
            raise ValueError(f"Unsupported model: {model_name}")
        
        # Custom classification head
        self.classifier = nn.Sequential(
            nn.Linear(num_features, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        
        # Freeze/unfreeze strategy
        if freeze_backbone:
            self._freeze_backbone(num_unfreeze_layers)
    
    def _freeze_backbone(self, num_unfreeze_layers: int):
        """
        Freeze backbone layers except last num_unfreeze_layers.
        
        Args:
            num_unfreeze_layers: Number of layers to keep trainable from the end
        """
        # Freeze all layers first
        for param in self.base_model.parameters():
            param.requires_grad = False
        
        # Unfreeze last N layers
        if num_unfreeze_layers > 0:
            layers = list(self.base_model.children())
            for layer in layers[-num_unfreeze_layers:]:
                for param in layer.parameters():
                    param.requires_grad = True
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor (batch, 3, 244, 244)
            
        Returns:
            Output tensor (batch, 1) with sigmoid activation
        """
        features = self.base_model(x)
        output = self.classifier(features)
        return output
    
    def get_num_parameters(self) -> Dict[str, int]:
        """Get parameter counts."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'total': total_params,
            'trainable': trainable_params,
            'frozen': total_params - trainable_params
        }


def create_model(
    model_type: str = 'custom',
    model_name: Optional[str] = None,
    pretrained: bool = True,
    **kwargs
) -> nn.Module:
    """
    Factory function to create flood detection models.
    
    Args:
        model_type: 'custom' or 'pretrained'
        model_name: For pretrained: 'resnet50', 'efficientnet_b0', 'mobilenet_v2'
        pretrained: Load ImageNet weights (for pretrained models)
        **kwargs: Additional model-specific arguments
        
    Returns:
        Initialized model
        
    Examples:
        >>> # Custom CNN
        >>> model = create_model('custom')
        
        >>> # ResNet50 pretrained
        >>> model = create_model('pretrained', model_name='resnet50')
        
        >>> # EfficientNet with frozen backbone
        >>> model = create_model('pretrained', model_name='efficientnet_b0', 
        ...                      freeze_backbone=True)
    """
    if model_type == 'custom':
        return CustomFloodCNN(**kwargs)
    
    elif model_type == 'pretrained':
        if model_name is None:
            raise ValueError("model_name required for pretrained models")
        return PretrainedFloodClassifier(
            model_name=model_name,
            pretrained=pretrained,
            **kwargs
        )
    
    else:
        raise ValueError(f"Unknown model_type: {model_type}")


if __name__ == "__main__":
    # Test models
    print("=" * 70)
    print("Testing Flood Detection Models")
    print("=" * 70)
    
    # Test input
    batch_size = 4
    test_input = torch.randn(batch_size, 3, 244, 244)
    
    # Test Custom CNN
    print("\n1. Custom CNN")
    print("-" * 70)
    custom_model = create_model('custom')
    output = custom_model(test_input)
    print(f"Input shape: {test_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Parameters: {custom_model.get_num_parameters():,}")
    print(f"Model size: {custom_model.get_num_parameters() * 4 / (1024**2):.2f} MB")
    
    # Test ResNet50
    print("\n2. ResNet50 (Pretrained)")
    print("-" * 70)
    resnet_model = create_model('pretrained', model_name='resnet50', pretrained=False)
    output = resnet_model(test_input)
    params = resnet_model.get_num_parameters()
    print(f"Input shape: {test_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Total parameters: {params['total']:,}")
    print(f"Trainable parameters: {params['trainable']:,}")
    print(f"Model size: {params['total'] * 4 / (1024**2):.2f} MB")
    
    # Test EfficientNet-B0
    print("\n3. EfficientNet-B0 (Pretrained)")
    print("-" * 70)
    effnet_model = create_model('pretrained', model_name='efficientnet_b0', pretrained=False)
    output = effnet_model(test_input)
    params = effnet_model.get_num_parameters()
    print(f"Input shape: {test_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Total parameters: {params['total']:,}")
    print(f"Trainable parameters: {params['trainable']:,}")
    print(f"Model size: {params['total'] * 4 / (1024**2):.2f} MB")
    
    # Test MobileNetV2
    print("\n4. MobileNetV2 (Pretrained)")
    print("-" * 70)
    mobilenet_model = create_model('pretrained', model_name='mobilenet_v2', pretrained=False)
    output = mobilenet_model(test_input)
    params = mobilenet_model.get_num_parameters()
    print(f"Input shape: {test_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Total parameters: {params['total']:,}")
    print(f"Trainable parameters: {params['trainable']:,}")
    print(f"Model size: {params['total'] * 4 / (1024**2):.2f} MB")
    
    print("\n" + "=" * 70)
    print("✅ All models tested successfully!")
    print("=" * 70)
