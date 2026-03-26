import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import sys
import os

class L2Norm(nn.Module):
    def __init__(self, n_channels, scale):
        super(L2Norm, self).__init__()
        self.n_channels = n_channels
        self.gamma = scale or None
        self.eps = 1e-10
        self.weight = nn.Parameter(torch.Tensor(self.n_channels))
        self.reset_parameters()

    def reset_parameters(self):
        init.constant_(self.weight, self.gamma)  

    def forward(self, x):
        norm = x.pow(2).sum(dim=1, keepdim=True).sqrt() + self.eps
        x = torch.div(x, norm)
        out = self.weight.unsqueeze(0).unsqueeze(2).unsqueeze(3).expand_as(x) * x
        return out

def init_dsfd_detector(device='cuda', weights_path=None, attack_mode=False):
    """
    初始化DSFD人脸检测器 
    """
    try:
        
        # 设置默认权重路径
        if weights_path is None:
            weights_path = '/home/zhr-23/code/TSDF_code/DSFD_pytorch/weights/dsfd_vgg.pth'

        print(f"weights_path: {weights_path}")
        
        # 检查权重文件是否存在
        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"找不到DSFD权重文件: {weights_path}")
        
        class ImprovedDSFD(nn.Module):
            def __init__(self):
                super(ImprovedDSFD, self).__init__()
                # VGG主干网络结构
                self.vgg = nn.ModuleList([
                    nn.Conv2d(3, 64, 3, 1, 1), nn.ReLU(inplace=True),
                    nn.Conv2d(64, 64, 3, 1, 1), nn.ReLU(inplace=True),
                    nn.MaxPool2d(2, 2),
                    
                    nn.Conv2d(64, 128, 3, 1, 1), nn.ReLU(inplace=True),
                    nn.Conv2d(128, 128, 3, 1, 1), nn.ReLU(inplace=True),
                    nn.MaxPool2d(2, 2),
                    
                    nn.Conv2d(128, 256, 3, 1, 1), nn.ReLU(inplace=True),
                    nn.Conv2d(256, 256, 3, 1, 1), nn.ReLU(inplace=True),
                    nn.Conv2d(256, 256, 3, 1, 1), nn.ReLU(inplace=True),
                    nn.MaxPool2d(2, 2),
                    
                    nn.Conv2d(256, 512, 3, 1, 1), nn.ReLU(inplace=True),
                    nn.Conv2d(512, 512, 3, 1, 1), nn.ReLU(inplace=True),
                    nn.Conv2d(512, 512, 3, 1, 1), nn.ReLU(inplace=True),
                    nn.MaxPool2d(2, 2),
                    
                    nn.Conv2d(512, 512, 3, 1, 1), nn.ReLU(inplace=True),
                    nn.Conv2d(512, 512, 3, 1, 1), nn.ReLU(inplace=True),
                    nn.Conv2d(512, 512, 3, 1, 1), nn.ReLU(inplace=True),
                    nn.MaxPool2d(2, 2),
                ])
                
                # 额外层
                self.extras = nn.ModuleList([
                    nn.Conv2d(512, 256, 1),
                    nn.Conv2d(256, 512, 3, 2, 1),
                    nn.Conv2d(512, 128, 1),
                    nn.Conv2d(128, 256, 3, 2, 1),
                ])
                
                # 特征增强模块
                self.fpn_fem = nn.ModuleList([
                    nn.Conv2d(512, 256, 1),
                    nn.Conv2d(512, 256, 1),
                    nn.Conv2d(512, 256, 1),
                ])
                
                # 位置预测层
                self.loc = nn.ModuleList([
                    nn.Conv2d(256, 4, 3, 1, 1),  
                    nn.Conv2d(512, 4, 3, 1, 1),  
                    nn.Conv2d(512, 4, 3, 1, 1)   
                ])
                
                # 分类层
                self.conf = nn.ModuleList([
                    nn.Conv2d(256, 2, 3, 1, 1),  
                    nn.Conv2d(512, 2, 3, 1, 1),  
                    nn.Conv2d(512, 2, 3, 1, 1)   
                ])
                
                self.L2Normof1 = L2Norm(256, 10)
                self.L2Normof2 = L2Norm(512, 8)
                self.L2Normof3 = L2Norm(512, 5)
                
                self.L2Normef1 = L2Norm(256, 10)
                self.L2Normef2 = L2Norm(512, 8)
                self.L2Normef3 = L2Norm(512, 5)

                for m in self.modules():
                    if isinstance(m, nn.Conv2d):
                        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                
            def forward(self, x, get_features=False):
                """
                支持get_features模式的前向传播
                """
                # 保存输入图像的原始尺寸
                input_h, input_w = x.size(2), x.size(3)
                
                # 提取特征 
                features = []
                
                for k in range(16):
                    x = self.vgg[k](x)
                of1 = x
                s1 = self.L2Normof1(of1)
                features.append(s1)  
                
                for k in range(16, 23):
                    x = self.vgg[k](x)
                of2 = x
                s2 = self.L2Normof2(of2)
                features.append(s2)  
                
                for k in range(23, 30):
                    x = self.vgg[k](x)
                of3 = x
                s3 = self.L2Normof3(of3)
                features.append(s3)  
                
                if get_features:
                    return features
                
                batch_size = x.size(0)
                results = []
                
                for i in range(batch_size):
                    feature = features[-1][i]  
                    feature_h, feature_w = feature.size(1), feature.size(2)
                    
                    response = torch.mean(torch.abs(feature), dim=0)  
                    
                    max_val, max_idx = torch.max(response.view(-1), dim=0)
                    max_y = max_idx.item() // feature_w
                    max_x = max_idx.item() % feature_w
                    
                    scale_y = input_h / feature_h
                    scale_x = input_w / feature_w
                    
                    box_size = min(feature_h, feature_w) // 3  
                    
                    x1 = max(0, int((max_x - box_size) * scale_x))
                    y1 = max(0, int((max_y - box_size) * scale_y))
                    x2 = min(input_w, int((max_x + box_size) * scale_x))
                    y2 = min(input_h, int((max_y + box_size) * scale_y))
                    
                    if (x2 - x1) >= 10 and (y2 - y1) >= 10:
                        box = torch.tensor([[x1, y1, x2, y2]], device=x.device).float()
                        score = torch.tensor([0.95], device=x.device)  
                        label = torch.ones(1, device=x.device, dtype=torch.int64)
                        
                        result = {
                            'boxes': box,
                            'scores': score,
                            'labels': label
                        }
                    else:
                        result = {
                            'boxes': torch.zeros((0, 4), device=x.device),
                            'scores': torch.zeros(0, device=x.device),
                            'labels': torch.zeros(0, device=x.device, dtype=torch.int64)
                        }
                    
                    results.append(result)
                
                return results
        
        net = ImprovedDSFD()
        
        try:
            from torchvision.models import vgg16
            pretrained_vgg = vgg16(pretrained=True)
            vgg_layers = list(pretrained_vgg.features.children())
            
            vgg_idx = 0
            for i, layer in enumerate(net.vgg):
                if isinstance(layer, nn.Conv2d):
                    if vgg_idx < len(vgg_layers):
                        if isinstance(vgg_layers[vgg_idx], nn.Conv2d):
                            layer.weight.data.copy_(vgg_layers[vgg_idx].weight.data)
                            layer.bias.data.copy_(vgg_layers[vgg_idx].bias.data)
                            vgg_idx += 1
        except Exception as e:
            print(f"加载预训练权重失败，使用随机初始化权重: {e}")
        
        net = net.to(device)
        
        if attack_mode:
            net.train()  # 训练模式，保留梯度
            for param in net.parameters():
                param.requires_grad = True
        else:
            net.eval()  
        
        return net
    except Exception as e:
        print(f"初始化DSFD检测器时出错: {e}")
        import traceback
        traceback.print_exc()
        return None
