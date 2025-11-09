import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights
import warnings
warnings.filterwarnings("ignore")

class BaselineFNN(nn.Module):
    def __init__(self, input_size, num_classes):
        super(BaselineFNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, num_classes)
        
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
    
class EnhancedFNN(nn.Module):
    def __init__(self, input_size, num_classes):
        super(EnhancedFNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 1680)
        self.fc2 = nn.Linear(1680, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 64)
        self.fc5 = nn.Linear(64, num_classes)
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.25)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.dropout(self.relu(self.fc2(x)))
        x = self.dropout(self.relu(self.fc3(x)))
        x = self.dropout(self.relu(self.fc4(x)))
        x = self.fc5(x)
        return x
    
    
class BaselineCNN(nn.Module):
    def __init__(self, input_size, num_classes):
        super(BaselineCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=8, padding=0)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, padding=0)
        
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.fc1 = nn.Linear(64 * 28 * 28, 128)
        self.fc2 = nn.Linear(128, num_classes)
        
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
    
class MediumCNN(nn.Module):
    def __init__(self, input_size, num_classes):
        super(MediumCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=8, padding=0)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, padding=0)
        
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.gap = nn.AdaptiveAvgPool2d((1,1))
        
        self.fc1 = nn.Linear(64, 16)
        self.fc2 = nn.Linear(16, num_classes)
        
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x)
        return x
    
    
class EnhancedCNN(nn.Module):
    def __init__(self, input_size, num_classes):
        super(EnhancedCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=8, padding=0)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, padding=0)
        
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.gap = nn.AdaptiveAvgPool2d((2,2))
        
        self.fc1 = nn.Linear(256, 16)
        self.fc2 = nn.Linear(16, num_classes)
        
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(0.7)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x)
        return x
    
    
class se_block(nn.Module):
    """
    Squeeze-and-excitation block for channel-wise feature recalibration.
    """
    def __init__(self, channel, reduction=16):
        super(se_block, self).__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        
        # Squeeze
        y = self.gap(x).view(b, c) 
        
        # Excitation
        z = self.fc(y).view(b, c, 1, 1)

        out = x * z.expand_as(x)
        return out
    
class seCNN(nn.Module):
    def __init__(self, input_size, num_classes):
        super(seCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=8, padding=0)
        self.bn1 = nn.BatchNorm2d(32)
        self.se1 = se_block(32, reduction=8)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, padding=0)
        self.bn2 = nn.BatchNorm2d(64)
        self.se2 = se_block(64, reduction=8)
        
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.gap = nn.AdaptiveAvgPool2d((1, 1)) 
        
        self.fc1 = nn.Linear(64, 16)
        self.fc2 = nn.Linear(16, num_classes)
        
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.se1(self.pool(self.relu(self.bn1(self.conv1(x)))))
        x = self.se2(self.pool(self.relu(self.bn2(self.conv2(x)))))
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x)
        return x
    
    
class DFFN(nn.Module):
    def __init__(self, input_size, num_classes):
        super(DFFN, self).__init__()
        # low level
        self.conv_low = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.bn_low = nn.BatchNorm2d(16)
        self.pool_low = nn.MaxPool2d(kernel_size=2, stride=2)
        self.gap_low = nn.AdaptiveAvgPool2d((1, 1))

        # high level
        self.conv_high_1 = nn.Conv2d(1, 32, kernel_size=8, padding=0)
        self.bn_high_1 = nn.BatchNorm2d(32)
        self.pool_high_1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # second high level
        self.conv_high_2 = nn.Conv2d(32, 64, kernel_size=5, padding=0)
        self.bn_high_2 = nn.BatchNorm2d(64)
        self.gap_high = nn.AdaptiveAvgPool2d((1, 1))

        # global level
        self.fc_global_1 = nn.Linear(128 * 128, 128)
        self.bn_global_1 = nn.BatchNorm1d(128)
        self.dropout_global = nn.Dropout(0.5)

        # fusion and classification
        self.dropout_fusion = nn.Dropout(0.5)
        self.fc_fusion = nn.Linear(16 + 64 + 128 , 64)
        self.fc_final = nn.Linear(64, num_classes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x_low = self.pool_low(self.relu(self.bn_low(self.conv_low(x))))
        x_low = self.gap_low(x_low)
        x_low = x_low.view(x_low.size(0), -1)

        x_high = self.pool_high_1(self.relu(self.bn_high_1(self.conv_high_1(x))))
        x_high = self.relu(self.bn_high_2(self.conv_high_2(x_high)))
        x_high = self.gap_high(x_high)
        x_high = x_high.view(x_high.size(0), -1)

        x_global = x.view(x.size(0), -1)
        x_global = self.relu(self.bn_global_1(self.fc_global_1(x_global)))
        x_global = self.dropout_global(x_global)
        
        x_fused = torch.cat((x_low, x_high, x_global), dim=1)
        x_fused = self.dropout_fusion(self.relu(self.fc_fusion(x_fused)))
        output = self.fc_final(x_fused)
        return output
    
    
class AugmentedCNN(nn.Module):
    def __init__(self, input_size, num_classes):
        super(AugmentedCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=8, padding=0)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, padding=0)
        
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        
        self.fc1 = nn.Linear(64, 16) 
        self.fc2 = nn.Linear(16, num_classes)
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x)
        return x
    
    
class BaselineResnetTransfer(nn.Module):
    def __init__(self, input_size, num_classes):
        super(BaselineResnetTransfer, self).__init__()
        self.resnet = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        
        og_conv1 = self.resnet.conv1
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.resnet.conv1.weight.data = og_conv1.weight.data[:, 0, :, :].unsqueeze(1)
        
        num_feats = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_feats, num_classes)
        
        for name, param in self.resnet.named_parameters():
            if "fc" not in name:
                param.requires_grad = False
            
    def forward(self, x):
        return self.resnet(x)


class EnhancedResnetTransfer(nn.Module):
    def __init__(self, input_size, num_classes):
        super(EnhancedResnetTransfer, self).__init__()
        self.resnet = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        
        og_conv1 = self.resnet.conv1
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.resnet.conv1.weight.data = og_conv1.weight.data[:, 0, :, :].unsqueeze(1)
        
        num_feats = self.resnet.fc.in_features
        
        self.resnet.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_feats, num_classes)
        )
                
        for param in self.resnet.parameters():
            param.requires_grad = False
            
        for name, param in self.resnet.named_parameters():
            if "layer4" in name or "layer3" in name or "fc" in name:
                param.requires_grad = True

        for m in self.resnet.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.weight.requires_grad = True
                m.bias.requires_grad = True
            
    def forward(self, x):
        return self.resnet(x)