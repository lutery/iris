"""
Credits to https://github.com/CompVis/taming-transformers
"""

from dataclasses import dataclass
from typing import List

import torch
import torch.nn as nn


@dataclass
class EncoderDecoderConfig:
    resolution: int # 图像的大小
    in_channels: int # 通道数
    z_channels: int # 
    ch: int # 第一层卷机的通道数
    ch_mult: List[int] # 用于控制每个残差连接的输入输出通道数的倍率
    num_res_blocks: int # 有多少个残差连接
    attn_resolutions: List[int]
    out_ch: int
    dropout: float


class Encoder(nn.Module):
    def __init__(self, config: EncoderDecoderConfig) -> None:
        '''
        对应配置文件tokenizer
        '''
        super().__init__()
        self.config = config
        self.num_resolutions = len(config.ch_mult) # todo
        temb_ch = 0  # timestep embedding #channels # 这个的作用

        # downsampling 下采样
        self.conv_in = torch.nn.Conv2d(config.in_channels,
                                       config.ch,
                                       kernel_size=3,
                                       stride=1,
                                       padding=1)

        curr_res = config.resolution
        in_ch_mult = (1,) + tuple(config.ch_mult)
        self.down = nn.ModuleList()
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = config.ch * in_ch_mult[i_level]
            block_out = config.ch * config.ch_mult[i_level]
            for i_block in range(self.config.num_res_blocks):
                block.append(ResnetBlock(in_channels=block_in,
                                         out_channels=block_out,
                                         temb_channels=temb_ch,
                                         dropout=config.dropout))
                block_in = block_out
                if curr_res in config.attn_resolutions:
                    # 如果输出的特征图的尺寸是指定的尺寸，则添加注意力机制
                    attn.append(AttnBlock(block_in))
            # 下采样层
            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions - 1:
                # 如果不是最后一层，则添加下采样层
                # 下采样的倍率为2
                down.downsample = Downsample(block_in, with_conv=True)
                curr_res = curr_res // 2
            self.down.append(down)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=temb_ch,
                                       dropout=config.dropout)
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=temb_ch,
                                       dropout=config.dropout)
        

        # end
        self.norm_out = Normalize(block_in)
        # 最后一层输出z_channels个通道，看起来就是特征图的通道数
        # 后续也是基于这个做一个随机概率采样吧
        self.conv_out = torch.nn.Conv2d(block_in,
                                        config.z_channels,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''
        x: shape is (N*T, C, H, W) 观察数据
        '''

        temb = None  # timestep embedding

        # downsampling
        hs = [self.conv_in(x)] # 存储每一层的输出
        for i_level in range(self.num_resolutions): # 控制进行几次下采样
            for i_block in range(self.config.num_res_blocks): # 下采样之间ResNet层数
                # hs[-1] # 上一层的输出 temb： None
                h = self.down[i_level].block[i_block](hs[-1], temb)
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
                hs.append(h)
            if i_level != self.num_resolutions - 1:
                hs.append(self.down[i_level].downsample(hs[-1]))

        # middle 
        h = hs[-1] # 获取最新的一层的输出 h = （N*T, block_in, h(H/4), w(W/4)）
        h = self.mid.block_1(h, temb) # h shape不变 （N*T, block_in, h(H/4), w(W/4)）
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb) # h shape 不变 （N*T, block_in, h(H/4), w(W/4)）

        # end
        h = self.norm_out(h) # 进一步归一化
        h = nonlinearity(h) # 增强数据的拟合能力增强
        # 高维特征映射（block_in 通道）压缩到较低维度的潜在表示 todo 调试代码的时候看下是否向下压缩的，也就是数据量缩小
        # 生成潜在表示
        # 将处理后的特征转换为标准化的潜在空间表示
        # 这个潜在表示将被用于后续的任务（如重建、分类或策略学习）
        h = self.conv_out(h) # 后续可能会通过这个参数进行采样，采样后重新生成新的观察
        '''
        离散编码：通过 VQ (Vector Quantization) 或 tokenization 进一步处理
        世界模型输入：作为预测未来状态的世界模型的输入
        策略学习：为 actor-critic 策略提供观察表示
        '''
        return h # h shape is (N*T, z_channels, h(H/4), w(W/4))


class Decoder(nn.Module):
    '''
    如果没猜错，就是将特征图解码成图像的网络
    配置复用encoder的
    这里也要注意的是，这里面的已有的模型结构仅支持指定分辨率的上采样为64
    '''
    def __init__(self, config: EncoderDecoderConfig) -> None:
        super().__init__()
        self.config = config
        temb_ch = 0
        self.num_resolutions = len(config.ch_mult)

        # compute in_ch_mult, block_in and curr_res at lowest res
        in_ch_mult = (1,) + tuple(config.ch_mult)
        block_in = config.ch * config.ch_mult[self.num_resolutions - 1]
        curr_res = config.resolution // 2 ** (self.num_resolutions - 1)
        print(f"Tokenizer : shape of latent is {config.z_channels, curr_res, curr_res}.")

        # z to block_in
        self.conv_in = torch.nn.Conv2d(config.z_channels,
                                       block_in,
                                       kernel_size=3,
                                       stride=1,
                                       padding=1)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=temb_ch,
                                       dropout=config.dropout)
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=temb_ch,
                                       dropout=config.dropout)

        # upsamplin
        # 看起来这边的上采样就是增加了一个上采样层，
        # 其余的就是尺寸不变的特征提取
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = config.ch * config.ch_mult[i_level]
            for i_block in range(config.num_res_blocks + 1):
                block.append(ResnetBlock(in_channels=block_in,
                                         out_channels=block_out,
                                         temb_channels=temb_ch,
                                         dropout=config.dropout))
                block_in = block_out
                if curr_res in config.attn_resolutions:
                    attn.append(AttnBlock(block_in))
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = Upsample(block_in, with_conv=True)
                curr_res = curr_res * 2
            self.up.insert(0, up)  # prepend to get consistent order

        # end
        # 在这边就转换为3通道
        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv2d(block_in,
                                        config.out_ch,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        temb = None  # timestep embedding

        # z to block_in
        h = self.conv_in(z)

        # middle
        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)

        # upsampling
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.config.num_res_blocks + 1):
                h = self.up[i_level].block[i_block](h, temb)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            if i_level != 0:
                h = self.up[i_level].upsample(h)

        # end
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        # h = (N*T, out_ch, H, W)  # 输出的图像形状
        return h


def nonlinearity(x: torch.Tensor) -> torch.Tensor:
    # swish
    return x * torch.sigmoid(x)


def Normalize(in_channels: int) -> nn.Module:
    '''
    num_groups：要分成的组数
    num_channels：输入通道数
    eps：添加到分母中的小值，防止除零错误
    affine：如果为True，则添加可学习的仿射变换参数（gamma和beta）

    小批量训练：当GPU内存限制导致只能使用小批量（batch size < 8）时，GroupNorm的表现通常优于BatchNorm

    视觉任务：在计算机视觉任务中表现良好，尤其是图像分类、目标检测和分割

    序列长度不固定的任务：如NLP或视频处理，其中批量内样本长度可变

    对训练/推理一致性有高要求的应用：例如强化学习或增量学习

    高分辨率图像处理：在处理高分辨率图像时，GroupNorm节省内存且效果稳定
    '''
    return torch.nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)


class Upsample(nn.Module):
    def __init__(self, in_channels: int, with_conv: bool) -> None:
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''
        torch.nn.functional.interpolate(
                input,                # 输入张量
                size=None,            # 输出大小
                scale_factor=None,    # 缩放因子
                mode='nearest',       # 插值模式
                align_corners=None,   # 是否对齐角点
                recompute_scale_factor=None  # 是否重新计算缩放因子
        '''
        # todo 换成反卷积试试？进行对比，说是可能造成训练不稳定，生产的图片存在伪影
        x = torch.nn.functional.interpolate(x, scale_factor=2.0, mode="nearest")
        if self.with_conv:
            # 这里的卷积是为了进一步处理上采样后的特征图，保证生成的特征图更加平滑和一致
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    def __init__(self, in_channels: int, with_conv: bool) -> None:
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            # no asymmetric padding in torch conv, must do it ourselves
            self.conv = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=3,
                                        stride=2,
                                        padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.with_conv:
            # 使用卷积j卷积进行下采样
            pad = (0, 1, 0, 1)
            x = torch.nn.functional.pad(x, pad, mode="constant", value=0)
            x = self.conv(x)
        else:
            # 这里使用池化直接下采样
            x = torch.nn.functional.avg_pool2d(x, kernel_size=2, stride=2)
        return x


class ResnetBlock(nn.Module):
    def __init__(self, *, in_channels: int, out_channels: int = None, conv_shortcut: bool = False,
                 dropout: float, temb_channels: int = 512) -> None:
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut

        self.norm1 = Normalize(in_channels)
        self.conv1 = torch.nn.Conv2d(in_channels,
                                     out_channels,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1)
        if temb_channels > 0:
            self.temb_proj = torch.nn.Linear(temb_channels,
                                             out_channels)
        self.norm2 = Normalize(out_channels)
        self.dropout = torch.nn.Dropout(dropout)
        self.conv2 = torch.nn.Conv2d(out_channels,
                                     out_channels,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1)
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = torch.nn.Conv2d(in_channels,
                                                     out_channels,
                                                     kernel_size=3,
                                                     stride=1,
                                                     padding=1)
            else:
                self.nin_shortcut = torch.nn.Conv2d(in_channels,
                                                    out_channels,
                                                    kernel_size=1,
                                                    stride=1,
                                                    padding=0)

    def forward(self, x: torch.Tensor, temb: torch.Tensor) -> torch.Tensor:
        '''
        x: shape is (N*T, C, H, W) 观察数据 （N*T, block_in, h(H/4), w(W/4)）
        temb: shape is (N*T, temb_ch) 时间步嵌入 todo 但是目前还未发现有传入的地方，目前默认为None

        假设有传入temb，设计源自生成模型（特别是扩散模型）架构，其中时间步嵌入至关重要。在强化学习上下文中，temb 可以提供以下潜在价值：

        时序条件化：允许模型根据时间位置生成不同的表示，这在序列预测和规划中非常有用

        不确定性建模：在世界模型中，可以表示预测的不确定性随着时间推移的变化

        长期依赖性：帮助模型区分短期和长期记忆中的状态表示

        规划时域：在基于模型的强化学习中，可用于条件化不同时间尺度的规划
        '''
        h = x
        h = self.norm1(h)
        h = nonlinearity(h)
        h = self.conv1(h)

        if temb is not None:
            h = h + self.temb_proj(nonlinearity(temb))[:, :, None, None]

        h = self.norm2(h)
        h = nonlinearity(h)
        h = self.dropout(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)

        return x + h


class AttnBlock(nn.Module):
    '''
    todo 后续看注意力层是怎么工作的
    '''
    def __init__(self, in_channels: int) -> None:
        super().__init__()
        self.in_channels = in_channels

        self.norm = Normalize(in_channels) 
        self.q = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.k = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.v = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.proj_out = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=1,
                                        stride=1,
                                        padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''
        x:（N*T, block_in, h(H/4), w(W/4)）
        '''
        h_ = x
        h_ = self.norm(h_) # （N*T, block_in, h(H/4), w(W/4)）
        q = self.q(h_) # （N*T, block_in, h(H/4), w(W/4)）
        k = self.k(h_) # （N*T, block_in, h(H/4), w(W/4)）
        v = self.v(h_) # （N*T, block_in, h(H/4), w(W/4)）

        # compute attention
        b, c, h, w = q.shape # 这里的C已经经过了特征采样，所以这里的C应该有几百个
        q = q.reshape(b, c, h * w) # q shape is (N*T, block_in, h*H/4*w*W/4)
        q = q.permute(0, 2, 1)      # b,hw,c | q shape is (N*T, h*H/4*w*W/4, block_in)
        k = k.reshape(b, c, h * w)  # b,c,hw  | k shape is (N*T, block_in, h*H/4*w*W/4)
        # 是 PyTorch 中的批量矩阵乘法函数 (Batch Matrix Multiplication)，它对批次中的矩阵对执行矩阵乘法运算
        '''
        数学表达式：

        输入：两个 3D 张量 A 和 B，形状分别为 (b, n, m) 和 (b, m, p)
        输出：形状为 (b, n, p) 的张量 C，其中 C[i] = A[i] @ B[i]（@ 表示矩阵乘法）

        数学上，每个批次中，w_[b,i,j] 计算了位置 i 对位置 j 的注意力得分，通过计算 i 处的查询向量与 j 处的键向量的点积。
        '''
        w_ = torch.bmm(q, k)        # b,hw,hw    w[b,i,j]=sum_c q[b,i,c]k[b,c,j] | w shape is (N*T, h*H/4*w*W/4, h*H/4*w*W/4)
        # 查询-键相似度的缩放
        '''
        当特征维度 c 较大时，点积的方差也会增大。在没有缩放的情况下：

        方差 = O(c)
        这会导致 softmax 函数在高维度时产生极端的概率分布（接近 one-hot）

        缩放的目的是将方差稳定在 O(1)，使梯度更加稳定
        '''
        w_ = w_ * (int(c) ** (-0.5))
        # 产生每一个位置的注意力分数（这里用概率模拟）
        w_ = torch.nn.functional.softmax(w_, dim=2)

        # attend to values
        v = v.reshape(b, c, h * w) # v shape is (N*T, block_in, h*H/4*w*W/4)
        w_ = w_.permute(0, 2, 1)   # b,hw,hw (first hw of k, second of q) | w_ shape is (N*T, h*H/4*w*W/4, h*H/4*w*W/4)
        h_ = torch.bmm(v, w_)     # b, c,hw (hw of q) h_[b,c,j] = sum_i v[b,c,i] w_[b,i,j] | h shape is (N*T, block_in, h*H/4*w*W/4) 
        # 根据每个概率计算每一个位置的大小，将应该被忽略的变小，将要重视的变大
        h_ = h_.reshape(b, c, h, w) # h_ shape is (N*T, block_in, h(H/4), w(W/4))

        # 特征混合：融合各位置之间的注意力加权信息
        # 特征稳定：防止注意力机制引入过大的特征变化
        # 归一化控制：保持特征量级在合理范围    内
        h_ = self.proj_out(h_) # h_ shape is (N*T, block_in, h(H/4), w(W/4))    

        # 在进行残差连接之前，proj_out 确保注意力机制的输出与输入特征在同一特征空间，使残差连接有效
        # 防止特征崩塌：没有 proj_out，注意力机制可能导致特征分布过于集中
        # 可学习的特征混合：允许模型学习如何最佳地组合注意力输出
        # 增加模型容量：提供额外的参数来捕获复杂的特征关系
        # 残差路径稳定：确保残差连接两端的特征分布兼容
        return x + h_
