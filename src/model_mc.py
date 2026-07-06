import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from torchvision.models import resnet18, ResNet18_Weights,wide_resnet50_2,resnet34
import torchaudio


class Bottleneck(nn.Module):
    def __init__(self, inp, oup, stride, expansion):
        super(Bottleneck, self).__init__()
        self.connect = stride == 1 and inp == oup
        #
        self.conv = nn.Sequential(
            # pw
            nn.Conv2d(inp, inp * expansion, 1, 1, 0, bias=False),
            nn.BatchNorm2d(inp * expansion),
            nn.PReLU(inp * expansion),
            # nn.ReLU(inplace=True),

            # dw
            nn.Conv2d(inp * expansion, inp * expansion, 3, stride, 1, groups=inp * expansion, bias=False),
            nn.BatchNorm2d(inp * expansion),
            nn.PReLU(inp * expansion),
            # nn.ReLU(inplace=True),

            # pw-linear
            nn.Conv2d(inp * expansion, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
        )

    def forward(self, x):
        if self.connect:
            return x + self.conv(x)
        else:
            return self.conv(x)



class ConvBlock(nn.Module):
    def __init__(self, inp, oup, k, s, p, dw=False, linear=False):
        super(ConvBlock, self).__init__()
        self.linear = linear
        if dw:
            self.conv = nn.Conv2d(inp, oup, k, s, p, groups=inp, bias=False)
        else:
            self.conv = nn.Conv2d(inp, oup, k, s, p, bias=False)
        self.bn = nn.BatchNorm2d(oup)
        if not linear:
            self.prelu = nn.PReLU(oup)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.linear:
            return x
        else:
            return self.prelu(x)


Mobilefacenet_bottleneck_setting = [
    # t, c , n ,s
    [2, 128, 2, 2],
    [4, 128, 2, 2],
    [4, 128, 2, 2],
]

class TgramNet(nn.Module):
    def __init__(self, num_layer=3, mel_bins=128, win_len=1024, hop_len=512):
        super(TgramNet, self).__init__()
        # if "center=True" of stft, padding = win_len / 2
        self.conv_extrctor = nn.Conv1d(1, mel_bins, win_len, hop_len, win_len // 2, bias=False)
        self.conv_encoder = nn.Sequential(
            *[nn.Sequential(
                # 313(10) , 63(2), 126(4)
                nn.LayerNorm(313),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv1d(mel_bins, mel_bins, 3, 1, 1, bias=False),
            ) for idx in range(num_layer)])

    def forward(self, x):
        out = self.conv_extrctor(x)
        out = self.conv_encoder(out)
        return out



# MobileFaceNet BackBone
class MobileFaceNet(nn.Module):
    def __init__(self,
                 num_class,
                 bottleneck_setting=Mobilefacenet_bottleneck_setting,
                 in_channels=1,
              
                 ):
        super(MobileFaceNet, self).__init__()

        self.conv1 = ConvBlock(in_channels, 64, 3, 2, 1)
        self.dw_conv1 = ConvBlock(64, 64, 3, 1, 1, dw=True)

        self.inplanes = 64
        block = Bottleneck
        self.blocks = self._make_layer(block, bottleneck_setting)

        self.conv2 = ConvBlock(bottleneck_setting[-1][1], 512, 1, 1, 0)

     

        # 20(10), 4(2), 8(4)
        self.linear7 = ConvBlock(512, 512, (16, 20), 1, 0, dw=True, linear=True)
        
        self.linear1 = ConvBlock(512, 128, 1, 1, 0, linear=True)

        
        self.dropout = nn.Dropout(p=0.2)

        self.fc_out = nn.Linear(128, num_class)
        
        self.atten1 = CBAM(in_planes=64)
        self.atten2 = simam_module()
        
        # init
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, setting):
        layers = []
        for t, c, n, s in setting:
            for i in range(n):
                if i == 0:
                    layers.append(block(self.inplanes, c, s, t))
                else:
                    layers.append(block(self.inplanes, c, 1, t))
                self.inplanes = c

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)        # [B, 64, ...]
        
        x = self.atten1(x) # CBAM比较适合这个位置插入
        
        x = self.dw_conv1(x)     # [B, 64, ...]
        x = self.atten2(x) #
        x = self.blocks(x)       # [B, ..., ...]
        
        x = self.conv2(x)        # [B, 512, ...]
        
        x = self.linear7(x)
        
        x = self.linear1(x)

        feature = x.view(x.size(0), -1)
        feature = self.dropout(feature)   

        out = self.fc_out(feature)
        
        return out, feature




# Multi-Class SVDD
class ClassVDD_Tgram(nn.Module):
    """
    Backbone: MobileNetV2 (首层改 1 通道) -> GAP -> Linear(z_dim) -> Activation -> Linear(num_classes)
    - SVDD 使用激活后的中间特征 z
    - 分类使用最后线性层输出的 logits
    """

    def __init__(
            self,
            num_classes,
            device,
            z_dim: int = 128,
            eps_center: float = 0.01,
            leak_slope: float = 0.2,
            in_ch: int = 1,  # 输入通道，默认为 1
            mobilenet_width_mult: float = 1.0,
            bottleneck_setting = Mobilefacenet_bottleneck_setting,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.z_dim = z_dim
        self.device = device
        self.eps_center = eps_center
        self.leak_slope = leak_slope

        # ===== MobileNetV2 backbone=====
        self.mobilefacenet = MobileFaceNet(num_class=num_classes+z_dim,
                                           bottleneck_setting=bottleneck_setting,
                                           in_channels=2)
        # ===== TGram Feature Extractor =====
        self.tgramnet = TgramNet(mel_bins=128, win_len=1024, hop_len=512)
       
        self.register_buffer('c', torch.zeros(num_classes, z_dim))

        # Soft Edge
        self.nu = 0.1  # 0.05~0.2

        # Trainable Radius for each class
        self.R_raw = nn.Parameter(torch.full((num_classes,), -10.0))



    def forward(self, x_mel: torch.Tensor, x_wav: torch.Tensor,return_logits: bool = False):
        """
        x: (B,1,H,W)
        return:
          logits: [B, num_classes]
          z: 激活后的中间特征，shape [B, z_dim]
        """
        x_t = self.tgramnet(x_wav).unsqueeze(1)
        x = torch.cat((x_mel, x_t), dim=1)
      
       
        out, feature = self.mobilefacenet(x)

        # latent for svdd and logits for classification
        z = out[:, :self.z_dim]
        logits = out[:, self.z_dim:]
        return logits, z

    # ---------------------- initialize center ----------------------
    def set_c(self, dataloader):
        """
        随机初始化每类中心 c[k]，与标签无关；不扫描数据。
        仍保留原函数名与调用方式，方便兼容你现有训练代码。
        """
        c = torch.randn(self.num_classes, self.z_dim).to(self.device)
        # 可选：行归一化 + eps 保护
        # c = F.normalize(c, dim=1)
        eps = self.eps_center
        c[(c.abs() < eps) & (c < 0)] = -eps
        c[(c.abs() < eps) & (c > 0)] = eps
        if 'c' not in self._buffers:
            self.register_buffer('c', c)
        else:
            self.c.copy_(c)


    # ---------------------- Multi-Class SVDD Loss (Hard Edge) ----------------------
    def compute_loss(self, x_mel: torch.Tensor, x_wav: torch.Tensor,label: torch.Tensor) -> torch.Tensor:
        """
        DeepSVDD 损失：样本到其对应类别中心的距离平方；使用激活后的中间特征 z
        """
        if self.c is None:
            raise RuntimeError("Centers `self.c` not initialized. Call `set_c(dataloader)` first.")
        _,z = self.forward(x_mel,x_wav)              # [B, z_dim]
        cz = self.c[label]                   # [B, z_dim]（y: 0..K-1 的 LongTensor）
        loss_svdd = torch.mean(torch.sum((z - cz) ** 2, dim=1))
        return loss_svdd



    # ---------------------- Multi-Class SVDD Loss (Soft Edge) ----------------------
    def compute_soft_svdd_loss(self, x_mel: torch.Tensor, x_wav: torch.Tensor, label: torch.Tensor):
        """
        dist = ||z - c_y||^2
        R_y = softplus(R_raw[y]) >= 0
        loss = mean(R_y^2) + (1/nu) * mean( max(0, dist - R_y^2) )
        返回: (loss_soft, scores)；scores=dist-R_y^2（越大越异常）
        """
        if self.c is None:
            raise RuntimeError("Centers `self.c` not initialized. Call `set_c(dataloader)` first.")

        logits, z = self.forward(x_mel,x_wav)  # z: [B, z_dim]
        cz = self.c[label]  # [B, z_dim]
        dist = torch.sum((z - cz) ** 2, dim=1)  # [B]

        R = F.softplus(self.R_raw)  # [K]
        R_y = R[label]  # [B]
        scores = dist - R_y ** 2

        loss_R = torch.mean(R_y ** 2)
        loss_hinge = torch.mean(torch.clamp(scores, min=0.0))
        loss_soft = loss_R + (1.0 / self.nu) * loss_hinge
        return loss_soft, scores




    # ---------------------- Multi-Class SVDD Anomaly Score ----------------------
    def compute_anomaly_score(self, x_mel: torch.Tensor,x_wav: torch.Tensor,label: torch.Tensor) -> torch.Tensor:
        """
        异常分数：与所有类中心的距离平方取最小值（类无关评分）
        返回：shape [B]
        """
        if self.c is None:
            raise RuntimeError("Centers `self.c` not initialized. Call `set_c(dataloader)` first.")
        logits, z = self.forward(x_mel,x_wav)  # [B, z_dim]
        #d2 = torch.sum((z.unsqueeze(1) - self.c) ** 2, dim=2)  # [B, K]
        cz = self.c[label]
        score = torch.sum((z - cz) ** 2)

        #score, _ = torch.min(d2, dim=1)
        return score


    @torch.no_grad()
    def _l2_normalize(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(x, dim=1) if self.contrastive_l2_normalize else x





# Multi-Class SVDD mel spectrogram only 比采用tgram略微差一点
class ClassVDD_newmobile_mel_only(nn.Module):
    """
    Backbone: MobileNetV2 (首层改 1 通道) -> GAP -> Linear(z_dim) -> Activation -> Linear(num_classes)
    - SVDD 使用激活后的中间特征 z
    - 分类使用最后线性层输出的 logits
    """

    def __init__(
            self,
            num_classes,
            device,
            nu: float = 0.2,
            z_dim: int = 128,
            eps_center: float = 0.01,
            leak_slope: float = 0.2,
            in_ch: int = 1,  # 输入通道，默认为 1
            mobilenet_width_mult: float = 1.0,
            bottleneck_setting = Mobilefacenet_bottleneck_setting,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.z_dim = z_dim
        self.device = device
        self.eps_center = eps_center
        self.leak_slope = leak_slope

        # ===== MobileNetV2 backbone=====
        self.mobilefacenet = MobileFaceNet(num_class=num_classes+z_dim,
                                           bottleneck_setting=bottleneck_setting,
                                           in_channels=1
                                           )
        
        
        
       
        # ===== TGram Feature Extractor =====
        # self.tgramnet = TgramNet(mel_bins=256, win_len=1024, hop_len=512)
        
      
        
        self.register_buffer('c', torch.zeros(num_classes, z_dim))

        # Soft Edge
        self.nu = nu  # 0.05~0.2

        # Trainable Radius for each class
        self.R_raw = nn.Parameter(torch.full((num_classes,), -10.0))



    def forward(self, x_mel: torch.Tensor, x_wav: torch.Tensor,return_logits: bool = False):
        """
        x: (B,1,H,W)
        return:
          logits: [B, num_classes]
          z: 激活后的中间特征，shape [B, z_dim]
        """
        
        
        out, feature = self.mobilefacenet(x_mel)

        # latent for svdd and logits for classification
        z = out[:, :self.z_dim]
        logits = out[:, self.z_dim:]
        return logits, z

    # ---------------------- initialize center ----------------------
    def set_c(self, dataloader):
        """
        随机初始化每类中心 c[k]，与标签无关；不扫描数据。
        仍保留原函数名与调用方式，方便兼容你现有训练代码。
        """
        c = torch.randn(self.num_classes, self.z_dim).to(self.device)
        # 可选：行归一化 + eps 保护
        # c = F.normalize(c, dim=1)
        eps = self.eps_center
        c[(c.abs() < eps) & (c < 0)] = -eps
        c[(c.abs() < eps) & (c > 0)] = eps
        if 'c' not in self._buffers:
            self.register_buffer('c', c)
        else:
            self.c.copy_(c)


    # ---------------------- Multi-Class SVDD Loss (Hard Edge) ----------------------
    def compute_loss(self, x_mel: torch.Tensor, x_wav: torch.Tensor,label: torch.Tensor) -> torch.Tensor:
        """
        DeepSVDD 损失：样本到其对应类别中心的距离平方；使用激活后的中间特征 z
        """
        if self.c is None:
            raise RuntimeError("Centers `self.c` not initialized. Call `set_c(dataloader)` first.")
        _,z = self.forward(x_mel,x_wav)              # [B, z_dim]
        cz = self.c[label]                   # [B, z_dim]（y: 0..K-1 的 LongTensor）
        loss_svdd = torch.mean(torch.sum((z - cz) ** 2, dim=1))
        return loss_svdd

    # -------------------- One-Class SVDD Loss (Hard Edge) -------------------
    def compute_oneclass_loss(self, x_mel: torch.Tensor, x_wav: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        """
        DeepSVDD 损失：样本到其对应类别中心的距离平方；使用激活后的中间特征 z
        """
        if self.c is None:
            raise RuntimeError("Centers `self.c` not initialized. Call `set_c(dataloader)` first.")
        _, z = self.forward(x_mel,x_wav)  # [B, z_dim]
        cz = torch.mean(self.c,dim=0,keepdim=True)  # [B, z_dim]（y: 0..K-1 的 LongTensor）
        loss_svdd = torch.mean(torch.sum((z - cz) ** 2, dim=1))
        return loss_svdd


    # ---------------------- Multi-Class SVDD Loss (Soft Edge) ----------------------
    def compute_soft_svdd_loss(self, x_mel: torch.Tensor, x_wav: torch.Tensor, label: torch.Tensor):
        """
        dist = ||z - c_y||^2
        R_y = softplus(R_raw[y]) >= 0
        loss = mean(R_y^2) + (1/nu) * mean( max(0, dist - R_y^2) )
        返回: (loss_soft, scores)；scores=dist-R_y^2（越大越异常）
        """
        if self.c is None:
            raise RuntimeError("Centers `self.c` not initialized. Call `set_c(dataloader)` first.")

        logits, z = self.forward(x_mel,x_wav)  # z: [B, z_dim]
        cz = self.c[label]  # [B, z_dim]
        dist = torch.sum((z - cz) ** 2, dim=1)  # [B]

        R = F.softplus(self.R_raw)  # [K]
        R_y = R[label]  # [B]
        scores = dist - R_y ** 2

        loss_R = torch.mean(R_y ** 2)
        loss_hinge = torch.mean(torch.clamp(scores, min=0.0))
        loss_soft = loss_R + (1.0 / self.nu) * loss_hinge
        return loss_soft, scores



    # ---------------------- Classification Loss  ----------------------
    def compute_classification_loss(self, x_mel: torch.Tensor, x_wav: torch.Tensor ,label: torch.Tensor):
        """
        交叉熵分类损失；使用最后线性层输出 logits
        返回：loss_cls, logits
        """
        logits, z = self.forward(x_mel, x_wav, return_logits=True)
        loss_cls = F.cross_entropy(logits, label)
        return loss_cls, logits

    # ---------------------- One-Class SVDD Anomaly Score ----------------------
    def compute_oneclass_anomaly_score(self, x_mel: torch.Tensor, x_wav: torch.Tensor,label: torch.Tensor) -> torch.Tensor:
        """
        异常分数：与所有类中心的距离平方取最小值（类无关评分）
        返回：shape [B]
        """
        if self.c is None:
            raise RuntimeError("Centers `self.c` not initialized. Call `set_c(dataloader)` first.")
        logits, z = self.forward(x_mel,x_wav)  # [B, z_dim]
        # d2 = torch.sum((z.unsqueeze(1) - self.c) ** 2, dim=2)  # [B, K]
        cz = torch.mean(self.c,dim=0,keepdim=True)
        score = torch.sum((z - cz) ** 2, dim=1)

        # score, _ = torch.min(d2, dim=1)
        return score

    # ---------------------- Multi-Class SVDD Anomaly Score ----------------------
    def compute_anomaly_score(self, x_mel: torch.Tensor,x_wav: torch.Tensor,label: torch.Tensor) -> torch.Tensor:
        """
        异常分数：与所有类中心的距离平方取最小值（类无关评分）
        返回：shape [B]
        """
        if self.c is None:
            raise RuntimeError("Centers `self.c` not initialized. Call `set_c(dataloader)` first.")
        logits, z = self.forward(x_mel,x_wav)  # [B, z_dim]
        #d2 = torch.sum((z.unsqueeze(1) - self.c) ** 2, dim=2)  # [B, K]
        cz = self.c[label]
        score = torch.sum((z - cz) ** 2, dim=1)

        #score, _ = torch.min(d2, dim=1)
        return score

    
    # ---------------------- 考虑分类错误的异常分数计算 -----------------------
    def compute_anomaly_score_with_classification_weight(
            self,
            x_mel: torch.Tensor,
            x_wav: torch.Tensor,
            label: torch.Tensor,
            gamma: float = 1.0,  # 映射陡峭度：越大，偏离越敏感
            coef_max: float = 10.0,  # 系数上限，防极端值
            eps: float = 1e-12
    ) -> torch.Tensor:
        """
        若最近中心==真实标签: score = ||z - c_label||^2
        否则: score = ||z - c_label||^2 * alpha, 其中 alpha = (p_true + eps)^(-gamma) ∈ [1, coef_max]
        返回: [B]
        """
        if self.c is None:
            raise RuntimeError("Centers `self.c` not initialized. Call `set_c(dataloader)` first.")

        logits, z = self.forward(x_mel, x_wav)  # logits: [B,K], z: [B,D]
        # 与各中心的距离平方
        d2 = torch.sum((z.unsqueeze(1) - self.c) ** 2, dim=2)  # [B,K]
        pred = torch.argmin(d2, dim=1)  # [B]

        # 到真实类中心的距离（原逻辑的基准分数）
        d_true = torch.sum((z - self.c[label]) ** 2, dim=1)  # [B]

        # 用分类 logits 计算真类概率 p_true
        p = torch.softmax(logits, dim=1)  # [B,K]
        p_true = p.gather(1, label.view(-1, 1)).squeeze(1)  # [B]

        # 将 p_true -> 系数 alpha，保证 >=1，并加上上限避免爆炸
        alpha = torch.clamp((p_true + eps).pow(-gamma), min=1.0, max=coef_max)  # [B]

        # 仅在预测错误时应用放大系数
        mismatch = (pred != label)
        score = d_true.clone()
        score[mismatch] = score[mismatch] * alpha[mismatch]

        return score


    
    # ---------------------- Multi-Class SVDD  Anomaly Score (Soft)----------------------
    def soft_boundary_anomaly_score(self, x_mel: torch.Tensor, x_wav: torch.Tensor, label: torch.Tensor = None):
        """
        若提供 label：返回 dist - R_y^2 （[B]）
        若不提供 label：返回 min_k(dist_k - R_k^2) （[B]）
        """
        logits, z = self.forward(x_mel,x_wav)
        R = F.softplus(self.R_raw)  # [K]

        if label is not None:
            cz = self.c[label]  # [B, z_dim]
            dist = torch.sum((z - cz) ** 2, dim=1)  # [B]
            return dist - R[label] ** 2
        else:
            d2 = torch.sum((z.unsqueeze(1) - self.c) ** 2, dim=2)  # [B,K]
            scores = d2 - R.view(1, -1) ** 2  # [B,K]
            return torch.min(scores, dim=1)[0]


    # ---------------------- Classification Anomaly Score ----------------------
    def compute_classification_anomaly_score(self, x_mel: torch.Tensor, x_wav: torch.Tensor) -> torch.Tensor:
        """
        异常分数：与所有类中心的距离平方取最小值（类无关评分）
        返回：shape [B]
        """
        if self.c is None:
            raise RuntimeError("Centers `self.c` not initialized. Call `set_c(dataloader)` first.")
        logits, z = self.forward(x_mel, x_wav)  # [B, z_dim]

        return logits

    @torch.no_grad()
    def _l2_normalize(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(x, dim=1) if self.contrastive_l2_normalize else x



# 三种特征提取，mobilenet骨干
class ClassVDD_newmobile_mel_tgram_sinc(nn.Module):
    """
    Backbone: MobileNetV2 (首层改 1 通道) -> GAP -> Linear(z_dim) -> Activation -> Linear(num_classes)
    - SVDD 使用激活后的中间特征 z
    - 分类使用最后线性层输出的 logits
    """

    def __init__(
            self,
            num_classes,
            device,
            nu: float = 0.2,
            z_dim: int = 128,
            eps_center: float = 0.01,
            leak_slope: float = 0.2,
            in_ch: int = 1,  # 输入通道，默认为 1
            mobilenet_width_mult: float = 1.0,
            bottleneck_setting=Mobilefacenet_bottleneck_setting,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.z_dim = z_dim
        self.device = device
        self.eps_center = eps_center
        self.leak_slope = leak_slope

        # ===== MobileNetV2 backbone=====
        self.mobilefacenet = MobileFaceNet(num_class=num_classes + z_dim,
                                           bottleneck_setting=bottleneck_setting,
                                           in_channels=1
                                           )

        
        # ===== sinc Feature Extractor =====
        self.sincnet = SincConv_fast(
            out_channels=256,  # 滤波器数量
            kernel_size=1025,  # 每个滤波器长度
            sample_rate=16000,  # 采样率
            stride=512,
            padding=512,
            min_low_hz=20,)

       

        

        self.register_buffer('c', torch.zeros(num_classes, z_dim))

        # Soft Edge
        self.nu = nu  # 0.05~0.2

        # Trainable Radius for each class
        self.R_raw = nn.Parameter(torch.full((num_classes,), -10.0))
        
        #self.sfe = SFE(in_channels=1)

        
  

        
    def forward(self, x_mel: torch.Tensor, x_wav: torch.Tensor, return_logits: bool = False):
        """
        x: (B,1,H,W)
        return:
          logits: [B, num_classes]
          z: 激活后的中间特征，shape [B, z_dim]
        """
        
        x_sinc = self.sincnet(x_wav)
        x_sinc = x_sinc.unsqueeze(1)
       

        
        
        out, feature = self.mobilefacenet(x_sinc)

        # latent for svdd and logits for classification
        z = out[:, :self.z_dim]
        logits = out[:, self.z_dim:]
        return logits, z
        

    # ---------------------- initialize center ----------------------
    def set_c(self, dataloader):
        """
        随机初始化每类中心 c[k]，与标签无关；不扫描数据。
        仍保留原函数名与调用方式，方便兼容你现有训练代码。
        """
        c = torch.randn(self.num_classes, self.z_dim).to(self.device)
        # 可选：行归一化 + eps 保护
        # c = F.normalize(c, dim=1)
        eps = self.eps_center
        c[(c.abs() < eps) & (c < 0)] = -eps
        c[(c.abs() < eps) & (c > 0)] = eps
        if 'c' not in self._buffers:
            self.register_buffer('c', c)
        else:
            self.c.copy_(c)

    # ---------------------- Multi-Class SVDD Loss (Hard Edge) ----------------------
    def compute_loss(self, x_mel: torch.Tensor, x_wav: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        """
        DeepSVDD 损失：样本到其对应类别中心的距离平方；使用激活后的中间特征 z
        """
        if self.c is None:
            raise RuntimeError("Centers `self.c` not initialized. Call `set_c(dataloader)` first.")
        _, z = self.forward(x_mel, x_wav)  # [B, z_dim]
        cz = self.c[label]  # [B, z_dim]（y: 0..K-1 的 LongTensor）
        loss_svdd = torch.mean(torch.sum((z - cz) ** 2, dim=1))
        return loss_svdd

    # -------------------- One-Class SVDD Loss (Hard Edge) -------------------
    def compute_oneclass_loss(self, x_mel: torch.Tensor, x_wav: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        """
        DeepSVDD 损失：样本到其对应类别中心的距离平方；使用激活后的中间特征 z
        """
        if self.c is None:
            raise RuntimeError("Centers `self.c` not initialized. Call `set_c(dataloader)` first.")
        _, z = self.forward(x_mel, x_wav)  # [B, z_dim]
        cz = torch.mean(self.c, dim=0, keepdim=True)  # [B, z_dim]（y: 0..K-1 的 LongTensor）
        loss_svdd = torch.mean(torch.sum((z - cz) ** 2, dim=1))
        return loss_svdd

    # ---------------------- Multi-Class SVDD Loss (Soft Edge) ----------------------
    def compute_soft_svdd_loss(self, x_mel: torch.Tensor, x_wav: torch.Tensor, label: torch.Tensor):
        """
        dist = ||z - c_y||^2
        R_y = softplus(R_raw[y]) >= 0
        loss = mean(R_y^2) + (1/nu) * mean( max(0, dist - R_y^2) )
        返回: (loss_soft, scores)；scores=dist-R_y^2（越大越异常）
        """
        if self.c is None:
            raise RuntimeError("Centers `self.c` not initialized. Call `set_c(dataloader)` first.")

        logits, z = self.forward(x_mel, x_wav)  # z: [B, z_dim]
        cz = self.c[label]  # [B, z_dim]
        dist = torch.sum((z - cz) ** 2, dim=1)  # [B]

        R = F.softplus(self.R_raw)  # [K]
        R_y = R[label]  # [B]
        scores = dist - R_y ** 2

        loss_R = torch.mean(R_y ** 2)
        loss_hinge = torch.mean(torch.clamp(scores, min=0.0))
        loss_soft = loss_R + (1.0 / self.nu) * loss_hinge
        return loss_soft, scores

    # ---------------------- Classification Loss  ----------------------
    def compute_classification_loss(self, x_mel: torch.Tensor, x_wav: torch.Tensor, label: torch.Tensor):
        """
        交叉熵分类损失；使用最后线性层输出 logits
        返回：loss_cls, logits
        """
        logits, z = self.forward(x_mel, x_wav, return_logits=True)
        loss_cls = F.cross_entropy(logits, label)
        return loss_cls, logits

    # ---------------------- One-Class SVDD Anomaly Score ----------------------
    def compute_oneclass_anomaly_score(self, x_mel: torch.Tensor, x_wav: torch.Tensor,
                                       label: torch.Tensor) -> torch.Tensor:
        """
        异常分数：与所有类中心的距离平方取最小值（类无关评分）
        返回：shape [B]
        """
        if self.c is None:
            raise RuntimeError("Centers `self.c` not initialized. Call `set_c(dataloader)` first.")
        logits, z = self.forward(x_mel, x_wav)  # [B, z_dim]
        # d2 = torch.sum((z.unsqueeze(1) - self.c) ** 2, dim=2)  # [B, K]
        cz = torch.mean(self.c, dim=0, keepdim=True)
        score = torch.sum((z - cz) ** 2, dim=1)

        # score, _ = torch.min(d2, dim=1)
        return score

    # ---------------------- Multi-Class SVDD Anomaly Score ----------------------
    def compute_anomaly_score(self, x_mel: torch.Tensor, x_wav: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        """
        异常分数：与所有类中心的距离平方取最小值（类无关评分）
        返回：shape [B]
        """
        if self.c is None:
            raise RuntimeError("Centers `self.c` not initialized. Call `set_c(dataloader)` first.")
        logits, z = self.forward(x_mel, x_wav)  # [B, z_dim]
        # d2 = torch.sum((z.unsqueeze(1) - self.c) ** 2, dim=2)  # [B, K]
        cz = self.c[label]
        score = torch.sum((z - cz) ** 2,dim=1)

        # score, _ = torch.min(d2, dim=1)
        return score

    # ---------------------- 考虑分类错误的异常分数计算 -----------------------
    def compute_anomaly_score_with_classification_weight(
            self,
            x_mel: torch.Tensor,
            x_wav: torch.Tensor,
            label: torch.Tensor,
            gamma: float = 1.0,  # 映射陡峭度：越大，偏离越敏感
            coef_max: float = 10.0,  # 系数上限，防极端值
            eps: float = 1e-12
    ) -> torch.Tensor:
        """
        若最近中心==真实标签: score = ||z - c_label||^2
        否则: score = ||z - c_label||^2 * alpha, 其中 alpha = (p_true + eps)^(-gamma) ∈ [1, coef_max]
        返回: [B]
        """
        if self.c is None:
            raise RuntimeError("Centers `self.c` not initialized. Call `set_c(dataloader)` first.")

        logits, z = self.forward(x_mel, x_wav)  # logits: [B,K], z: [B,D]
        # 与各中心的距离平方
        d2 = torch.sum((z.unsqueeze(1) - self.c) ** 2, dim=2)  # [B,K]
        pred = torch.argmin(d2, dim=1)  # [B]

        # 到真实类中心的距离（原逻辑的基准分数）
        d_true = torch.sum((z - self.c[label]) ** 2, dim=1)  # [B]

        # 用分类 logits 计算真类概率 p_true
        p = torch.softmax(logits, dim=1)  # [B,K]
        p_true = p.gather(1, label.view(-1, 1)).squeeze(1)  # [B]

        # 将 p_true -> 系数 alpha，保证 >=1，并加上上限避免爆炸
        alpha = torch.clamp((p_true + eps).pow(-gamma), min=1.0, max=coef_max)  # [B]

        # 仅在预测错误时应用放大系数
        mismatch = (pred != label)
        score = d_true.clone()
        score[mismatch] = score[mismatch] * alpha[mismatch]

        return score

    # ---------------------- Multi-Class SVDD  Anomaly Score (Soft)----------------------
    def soft_boundary_anomaly_score(self, x_mel: torch.Tensor, x_wav: torch.Tensor, label: torch.Tensor = None):
        """
        若提供 label：返回 dist - R_y^2 （[B]）
        若不提供 label：返回 min_k(dist_k - R_k^2) （[B]）
        """
        logits, z = self.forward(x_mel, x_wav)
        R = F.softplus(self.R_raw)  # [K]

        if label is not None:
            cz = self.c[label]  # [B, z_dim]
            dist = torch.sum((z - cz) ** 2, dim=1)  # [B]
            return dist - R[label] ** 2
        else:
            d2 = torch.sum((z.unsqueeze(1) - self.c) ** 2, dim=2)  # [B,K]
            scores = d2 - R.view(1, -1) ** 2  # [B,K]
            return torch.min(scores, dim=1)[0]

    # ---------------------- Classification Anomaly Score ----------------------
    def compute_classification_anomaly_score(self, x_mel: torch.Tensor, x_wav: torch.Tensor) -> torch.Tensor:
        """
        异常分数：与所有类中心的距离平方取最小值（类无关评分）
        返回：shape [B]
        """
        if self.c is None:
            raise RuntimeError("Centers `self.c` not initialized. Call `set_c(dataloader)` first.")
        logits, z = self.forward(x_mel, x_wav)  # [B, z_dim]

        return logits

    @torch.no_grad()
    def _l2_normalize(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(x, dim=1) if self.contrastive_l2_normalize else x








# Sinc backbone
class SincConv_fast(nn.Module):
    """Sinc-based 1D convolution with softplus-constrained frequency parameters."""

    @staticmethod
    def to_mel(hz: np.ndarray) -> np.ndarray:
        return 2595 * np.log10(1 + hz / 700)

    @staticmethod
    def to_hz(mel: np.ndarray) -> np.ndarray:
        return 700 * (10 ** (mel / 2595) - 1)

    def __init__(
        self,
        out_channels: int, # 输出滤波器数量
        kernel_size: int, # 每个滤波器的长度
        sample_rate: int = 16000, # 输入信号的采样率
        in_channels: int = 1, # 输入通道数1
        stride: int = 1, # Conv1d 的步长
        padding: int = 0, # Conv1d 的填充长度。
        dilation: int = 1, # Conv1d 的膨胀系数。
        bias: bool = False,
        groups: int = 1,
        min_low_hz: float = 20.0,
        min_band_hz: float = 5.0,
    ):
        super().__init__()

        if in_channels != 1:
            raise ValueError(f"SincConv only supports one input channel (got in_channels={in_channels}).")

        self.out_channels = out_channels
        self.kernel_size = int(kernel_size) | 1          # 强制奇数长度
        self.sample_rate = int(sample_rate)
        self.stride, self.padding, self.dilation = stride, padding, dilation
        if bias:
            raise ValueError("SincConv does not support bias.")
        if groups != 1:
            raise ValueError("SincConv does not support groups.")

        self.min_low_hz = float(min_low_hz)
        self.min_band_hz = float(min_band_hz)

        # ===== 初始化滤波器频率参数 =====
        low_hz = 30.0
        high_hz = self.sample_rate / 2.0 - (self.min_low_hz + self.min_band_hz)

        # mel频率
        mel = np.linspace(self.to_mel(low_hz), self.to_mel(high_hz), out_channels + 1)

        # 对应mel的真实频率
        hz = self.to_hz(mel)

        # 低通频率变为可学习参数
        self.low_hz_ = nn.Parameter(torch.tensor(hz[:-1], dtype=torch.float32).view(-1, 1))

        # 带通频率也为可学习参数
        self.band_hz_ = nn.Parameter(torch.tensor(np.diff(hz), dtype=torch.float32).view(-1, 1))

        # Hamming 半窗与半时间轴
        half = self.kernel_size // 2
        n_lin = torch.linspace(0, half - 1, steps=half)
        window_half = 0.54 - 0.46 * torch.cos(2 * math.pi * n_lin / self.kernel_size)
        self.register_buffer("window_half", window_half) # window half是固定参数
        n = (self.kernel_size - 1) / 2.0
        t_half = 2 * math.pi * torch.arange(-n, 0) / self.sample_rate
        self.register_buffer("t_half", t_half.view(1, -1))
        self.register_buffer("filters", torch.empty(out_channels, 1, self.kernel_size))

        self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB(stype='power')
        
    def forward(self, waveforms: torch.Tensor) -> torch.Tensor:
        dtype, device = waveforms.dtype, waveforms.device
        t_half = self.t_half.to(dtype=dtype, device=device)
        window_half = self.window_half.to(dtype=dtype, device=device)

        # ===== softplus 约束 =====
        low = self.min_low_hz + F.softplus(self.low_hz_)              # 代替 abs
        high = torch.clamp(
            low + self.min_band_hz + F.softplus(self.band_hz_),      # 代替 abs
            min=self.min_low_hz,
            max=self.sample_rate / 2.0,
        )
        band = (high - low)[:, 0]

        f_times_t_low = torch.matmul(low, t_half)
        f_times_t_high = torch.matmul(high, t_half)

        denom = (t_half / 2.0).to(dtype=dtype, device=device)
        eps = torch.finfo(dtype).tiny
        denom = torch.where(torch.abs(denom) < eps, torch.full_like(denom, eps), denom)

        band_pass_left = ((torch.sin(f_times_t_high) - torch.sin(f_times_t_low)) / denom) * window_half
        band_pass_center = 2 * band.view(-1, 1)
        band_pass_right = torch.flip(band_pass_left, dims=[1])
        band_pass = torch.cat([band_pass_left, band_pass_center, band_pass_right], dim=1)

        band_safe = torch.where(band <= 0, torch.full_like(band, 1.0), band)
        band_pass = band_pass / (2 * band_safe[:, None])

        self.filters = band_pass.view(self.out_channels, 1, self.kernel_size)

        feature = F.conv1d(
            waveforms,
            self.filters,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            bias=None,
            groups=1,
        )
        # feature = self.amplitude_to_db(feature) 
        
        return feature




class LayerNorm(nn.Module):

    def __init__(self, features, eps=1e-6):
        super(LayerNorm,self).__init__()
        self.gamma = nn.Parameter(torch.ones(features))
        self.beta = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta





















# -----------------------------------------attention block -------------------------------------------------

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        # 共享权重的 MLP
        # 第一个全连接层将通道数减少到 C/ratio
        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, kernel_size=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        # 第二个全连接层恢复通道数到 C
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, kernel_size=1, bias=False)
        
        self.sigmoid = nn.Sigmoid()


        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
    def forward(self, x):
        # x: [B, C, H, W]
        
        # 1. 平均池化分支: [B, C, 1, 1]
        avg_out = self.fc2(self.relu(self.fc1(self.avg_pool(x))))
        
        # 2. 最大池化分支: [B, C, 1, 1]
        max_out = self.fc2(self.relu(self.fc1(self.max_pool(x))))
        
        # 3. 相加并激活: [B, C, 1, 1]
        # CBAM 原始论文中是将两个分支的结果相加
        out = avg_out + max_out
        
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=3):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        # 输入通道数为 2，因为我们将平均池化和最大池化的结果在通道维度拼接
        # 输出通道数为 1，生成单通道的空间注意力图
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: [B, C, H, W]
        
        # 1. 沿通道维度求平均: [B, 1, H, W]
        avg_out = torch.mean(x, dim=1, keepdim=True)
        
        # 2. 沿通道维度求最大值: [B, 1, H, W]
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        
        # 3. 拼接: [B, 2, H, W]
        x_concat = torch.cat([avg_out, max_out], dim=1)
        
        # 4. 卷积处理: [B, 1, H, W]
        # 使用大核卷积(如7x7)来捕获更大的感受野，从而更好地定位物体
        out = self.conv1(x_concat)
        
        return self.sigmoid(out)

class CBAM(nn.Module):
    def __init__(self, in_planes, ratio=4, kernel_size=7):
        super(CBAM, self).__init__()
        # 先通道注意力，后空间注意力 (Serial Connection)
        self.ca = ChannelAttention(in_planes, ratio)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        """
        :param x: Input feature map [B, C, H, W]
        :return: Refined feature map [B, C, H, W]
        """
        # 1. Channel Attention
        # 计算通道权重 [B, C, 1, 1] 并与原特征相乘
        ca_out = self.ca(x) * x
        
        # 2. Spatial Attention
        # 计算空间权重 [B, 1, H, W] 并与经过通道加权的特征相乘
        sa_out = self.sa(ca_out) * ca_out
        
        return sa_out



#--------------------- SimAM注意力 -----------------------

class simam_module(torch.nn.Module):
    def __init__(self, channels = None, e_lambda = 1e-4):
        super(simam_module, self).__init__()

        self.activaton = nn.Sigmoid()
        self.e_lambda = e_lambda


    def forward(self, x):

        b, c, h, w = x.size()
        
        n = w * h - 1

        x_minus_mu_square = (x - x.mean(dim=[2,3], keepdim=True)).pow(2)
        y = x_minus_mu_square / (4 * (x_minus_mu_square.sum(dim=[2,3], keepdim=True) / n + self.e_lambda)) + 0.5

        return x * self.activaton(y)



if __name__ == "__main__":
    # 创建模型实例
    model = ResNet18_Model(num_class=10, in_channels=1)

    # 测试前向传播
    input_tensor = torch.randn(4, 1, 128, 313)  # batch_size=4, 单通道，224x224
    output, features = model(input_tensor)

    print(f"输出形状：{output.shape}")  # [4, 10]
    print(f"特征形状：{features.shape}")  # [4, 512]







