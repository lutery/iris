"""
Credits to https://github.com/CompVis/taming-transformers
"""

from collections import namedtuple
import hashlib
import os
from pathlib import Path
import requests

import torch
import torch.nn as nn
from torchvision import models
from tqdm import tqdm


class LPIPS(nn.Module):
    # Learned perceptual metric
    '''
    看起来是一个经过特别训练的图像感知相似度度量网络
    在 IRIS 强化学习算法中的作用主要体现在图像感知相似
    '''
    def __init__(self, use_dropout: bool = True):
        super().__init__()
        # 看起来是一个图像感知相似度度量的网络
        self.scaling_layer = ScalingLayer()
        self.chns = [64, 128, 256, 512, 512]  # vg16 features
        # 预训练的VGG16网络，可能是用来提取图像特征的
        self.net = vgg16(pretrained=True, requires_grad=False)
        # todo 以下的作用
        self.lin0 = NetLinLayer(self.chns[0], use_dropout=use_dropout)
        self.lin1 = NetLinLayer(self.chns[1], use_dropout=use_dropout)
        self.lin2 = NetLinLayer(self.chns[2], use_dropout=use_dropout)
        self.lin3 = NetLinLayer(self.chns[3], use_dropout=use_dropout)
        self.lin4 = NetLinLayer(self.chns[4], use_dropout=use_dropout)
        # 加载已经训练好的vgg_lpips模型
        self.load_from_pretrained()
        # 本模型不求梯度，不训练
        for param in self.parameters():
            param.requires_grad = False

    def load_from_pretrained(self) -> None:
        ckpt = get_ckpt_path(name="vgg_lpips", root=Path.home() / ".cache/iris/tokenizer_pretrained_vgg")  # Download VGG if necessary
        self.load_state_dict(torch.load(ckpt, map_location=torch.device("cpu")), strict=False)

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # 现对输入和目标图像进行预处理
        in0_input, in1_input = (self.scaling_layer(input), self.scaling_layer(target))
        # 计算VGG特征
        outs0, outs1 = self.net(in0_input), self.net(in1_input)
        feats0, feats1, diffs = {}, {}, {}
        lins = [self.lin0, self.lin1, self.lin2, self.lin3, self.lin4]
        for kk in range(len(self.chns)):
            # 对每个特征通道进行处理
            feats0[kk], feats1[kk] = normalize_tensor(outs0[kk]), normalize_tensor(outs1[kk])
            # 对比input target之间的特征差异
            diffs[kk] = (feats0[kk] - feats1[kk]) ** 2

        # spatial_average 计算最后两个维度的均值
        # 这里lins是一个包含多个NetLinLayer的列表，每个输出的通道都是1
        res = [spatial_average(lins[kk].model(diffs[kk]), keepdim=True) for kk in range(len(self.chns))]
        val = res[0] # 这里仅仅只是为了初始化才分开的
        for i in range(1, len(self.chns)):
            val += res[i] # 这里将所有的特征差异均值进行累加
        
        # val shape (B, 1, 1, 1)
        return val


class ScalingLayer(nn.Module):
    '''
    在 IRIS 强化学习算法中的作用主要体现在图像感知相似度度量（LPIPS - Learned Perceptual Image Patch Similarity）
    感知相似度评估： 在基于模型的强化学习中，ScalingLayer作为LPIPS的组成部分，帮助评估生成图像与目标图像的感知相似度，这比传统的逐像素比较(如MSE)更符合人类视觉感知。

    训练信号优化： 通过标准化预处理，使感知损失能够提供更稳定、一致的训练信号，帮助模型生成更真实的图像预测。

    模型迁移适配： 将输入图像适配到与预训练VGG模型兼容的数据分布，使得强化学习系统能够有效利用迁移学习，借助在大规模数据集上预训练的视觉模型的能力。
    '''
    def __init__(self) -> None:
        super(ScalingLayer, self).__init__()
        self.register_buffer('shift', torch.Tensor([-.030, -.088, -.188])[None, :, None, None])
        self.register_buffer('scale', torch.Tensor([.458, .448, .450])[None, :, None, None])

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        '''
        这段代码对输入图像的RGB通道进行特定的标准化处理，将值映射到预训练VGG网络期望的分布范围内
        '''
        return (inp - self.shift) / self.scale


class NetLinLayer(nn.Module):
    """ A single linear layer which does a 1x1 conv """
    def __init__(self, chn_in: int, chn_out: int = 1, use_dropout: bool = False) -> None:
        super(NetLinLayer, self).__init__()
        layers = [nn.Dropout(), ] if (use_dropout) else []
        layers += [nn.Conv2d(chn_in, chn_out, 1, stride=1, padding=0, bias=False), ]
        self.model = nn.Sequential(*layers)


class vgg16(torch.nn.Module):
    def __init__(self, requires_grad: bool = False, pretrained: bool = True) -> None:
        super(vgg16, self).__init__()
        # 加载预训练的VGG16模型
        vgg_pretrained_features = models.vgg16(pretrained=pretrained).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        self.N_slices = 5
        '''
        VGG16 虽然名为"16层"，但这里的16指的是具有权重参数的层数（即卷积层和全连接层），而不是网络中的所有操作层。实际上，VGG16的特征提取部分（features）包含了：

        13个卷积层（带权重）
        5个最大池化层（无权重）
        13个ReLU激活函数（无权重）
        '''
        # 将预训练的VGG16模型的特征提取部分分成5个切片
        for x in range(4):
            self.slice1.add_module(str(x), vgg_pretrained_features[x]) # 索引0-3，对应初始的低级特征（边缘和纹理）
        for x in range(4, 9):
            self.slice2.add_module(str(x), vgg_pretrained_features[x]) # 索引4-8，对应第一个池化层后的特征
        for x in range(9, 16):
            self.slice3.add_module(str(x), vgg_pretrained_features[x]) # 索引9-15，对应更深层的中级特征
        for x in range(16, 23):
            self.slice4.add_module(str(x), vgg_pretrained_features[x]) # 对应高级特征
        for x in range(23, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x]) # 对应最深层的特征（语义级特征）
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        h = self.slice1(X)
        h_relu1_2 = h
        h = self.slice2(h)
        h_relu2_2 = h
        h = self.slice3(h)
        h_relu3_3 = h
        h = self.slice4(h)
        h_relu4_3 = h
        h = self.slice5(h)
        h_relu5_3 = h
        # 保存每一个部分提取的特征并返回
        vgg_outputs = namedtuple("VggOutputs", ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3', 'relu5_3'])
        out = vgg_outputs(h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3, h_relu5_3)
        return out


def normalize_tensor(x: torch.Tensor, eps: float = 1e-10) -> torch.Tensor:
    '''
    函数实现的是一种特征向量的 L2 归一化（也称为"单位向量归一化"或"球面归一化"）
    '''
    norm_factor = torch.sqrt(torch.sum(x ** 2, dim=1, keepdim=True))
    return x / (norm_factor + eps)


def spatial_average(x: torch.Tensor, keepdim: bool = True) -> torch.Tensor:
    return x.mean([2, 3], keepdim=keepdim) # 保持维度，变化为 B C 1 1


# ********************************************************************
# *************** Utilities to download pretrained vgg ***************
# ********************************************************************

# 如果无法下载则使用同目录下的vgg.pth文件
URL_MAP = {
    "vgg_lpips": "https://heibox.uni-heidelberg.de/f/607503859c864bc1b30b/?dl=1"
}


CKPT_MAP = {
    "vgg_lpips": "vgg.pth"
}


MD5_MAP = {
    "vgg_lpips": "d507d7349b931f0638a25a48a722f98a"
}


def download(url: str, local_path: str, chunk_size: int = 1024) -> None:
    os.makedirs(os.path.split(local_path)[0], exist_ok=True)
    with requests.get(url, stream=True) as r:
        total_size = int(r.headers.get("content-length", 0))
        with tqdm(total=total_size, unit="B", unit_scale=True) as pbar:
            with open(local_path, "wb") as f:
                for data in r.iter_content(chunk_size=chunk_size):
                    if data:
                        f.write(data)
                        pbar.update(chunk_size)


def md5_hash(path: str) -> str:
    with open(path, "rb") as f:
        content = f.read()
    return hashlib.md5(content).hexdigest()


def get_ckpt_path(name: str, root: str, check: bool = False) -> str:
    assert name in URL_MAP
    path = os.path.join(root, CKPT_MAP[name])
    if not os.path.exists(path) or (check and not md5_hash(path) == MD5_MAP[name]):
        print("Downloading {} model from {} to {}".format(name, URL_MAP[name], path))
        download(URL_MAP[name], path)
        md5 = md5_hash(path)
        assert md5 == MD5_MAP[name], md5
    return path
