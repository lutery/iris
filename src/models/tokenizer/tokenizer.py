"""
Credits to https://github.com/CompVis/taming-transformers
"""

from dataclasses import dataclass
from typing import Any, Tuple

from einops import rearrange
import torch
import torch.nn as nn

from dataset import Batch
from .lpips import LPIPS
from .nets import Encoder, Decoder
from utils import LossWithIntermediateLosses


@dataclass
class TokenizerEncoderOutput:
    z: torch.FloatTensor # 连续潜在表示 shape is (N, T, embed_dim, h(H/4), w(W/4))
    z_quantized: torch.FloatTensor # 量化潜在表示 通过从码本中查找 `tokens` 对应的向量得到 shape is (N, T, embed_dim, h(H/4), w(W/4))
    tokens: torch.LongTensor # 通过找到与 `z` 最近的码本项得到 shape is (N, T, H/4*W/4)


class Tokenizer(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, encoder: Encoder, decoder: Decoder, with_lpips: bool = True) -> None:
        '''
        参数对应配置文件：tokenizer.yaml
        看起来有点像观察的特征提取器和反编码器

        看起来就是将观察和特征进行编码特征提取以及反编码为特征图
        '''
        
        super().__init__()
        self.vocab_size = vocab_size
        self.encoder = encoder
        self.pre_quant_conv = torch.nn.Conv2d(encoder.config.z_channels, embed_dim, 1)
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.post_quant_conv = torch.nn.Conv2d(embed_dim, decoder.config.z_channels, 1)
        self.decoder = decoder
        self.embedding.weight.data.uniform_(-1.0 / vocab_size, 1.0 / vocab_size)
        self.lpips = LPIPS().eval() if with_lpips else None

    def __repr__(self) -> str:
        return "tokenizer"

    def forward(self, x: torch.Tensor, should_preprocess: bool = False, should_postprocess: bool = False) -> Tuple[torch.Tensor]:
        outputs = self.encode(x, should_preprocess)
        decoder_input = outputs.z + (outputs.z_quantized - outputs.z).detach()
        reconstructions = self.decode(decoder_input, should_postprocess)
        return outputs.z, outputs.z_quantized, reconstructions

    def compute_loss(self, batch: Batch, **kwargs: Any) -> LossWithIntermediateLosses:
        assert self.lpips is not None
        observations = self.preprocess_input(rearrange(batch['observations'], 'b t c h w -> (b t) c h w'))
        z, z_quantized, reconstructions = self(observations, should_preprocess=False, should_postprocess=False)

        # Codebook loss. Notes:
        # - beta position is different from taming and identical to original VQVAE paper
        # - VQVAE uses 0.25 by default
        beta = 1.0
        commitment_loss = (z.detach() - z_quantized).pow(2).mean() + beta * (z - z_quantized.detach()).pow(2).mean()

        reconstruction_loss = torch.abs(observations - reconstructions).mean()
        perceptual_loss = torch.mean(self.lpips(observations, reconstructions))

        return LossWithIntermediateLosses(commitment_loss=commitment_loss, reconstruction_loss=reconstruction_loss, perceptual_loss=perceptual_loss)

    def encode(self, x: torch.Tensor, should_preprocess: bool = False) -> TokenizerEncoderOutput:
        '''
        x: burnin_obs shape is (N, T, C, H, W);
        shoud_preprocess: 是否需要预处理 True;
        '''
        if should_preprocess:
            x = self.preprocess_input(x)
        shape = x.shape  # (..., C, H, W) (N, T, C, H, W);
        x = x.view(-1, *shape[-3:]) # 这里是将 N 和 T 展开成一个维度 shape is (N*T, C, H, W)
        z = self.encoder(x) # z in shape is (N*T, block_in, h(H/4), w(W/4)) | z out shape is h shape is (N*T, z_channels, h(H/4), w(W/4))
        z = self.pre_quant_conv(z) # z shape is (N*T, embed_dim, h(H/4), w(W/4))
        b, e, h, w = z.shape
        z_flattened = rearrange(z, 'b e h w -> (b h w) e') # 将 z 展平为 (b*h*w, e) 的形状，对比view的优势就是不用处理内存是否连续的情况
        # z_flattened shape is (N*T*H/4*W/4, embed_dim)
        # self.embedding.weight shape is (vocab_size, embed_dim) todo 怀疑vocab_size=N*T*H/4*W/4 测试
        # dist_to_embeddings 计算的是计算向量量化(Vector Quantization)中的欧氏距离平方，用于找到编码本(codebook)中与输入向量最相似的项
        # 计算公式为：||z_flattened||^2 + ||embedding.weight||^2 - 2 * z_flattened @ embedding.weight.t()
        dist_to_embeddings = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + torch.sum(self.embedding.weight**2, dim=1) - 2 * torch.matmul(z_flattened, self.embedding.weight.t())

        # 以上计算完成后：最近邻查找：接下来的 dist_to_embeddings.argmin(dim=-1) 找到最近的码本向量
        # 离散表示：将连续向量映射到离散空间中的索引（tokens）
        # 量化：用找到的最近码本向量替换原始向量，创建量化表示
        tokens = dist_to_embeddings.argmin(dim=-1) # 找到最近的码本向量索引，tokens shape is (N*T*H/4*W/4,)
        z_q = rearrange(self.embedding(tokens), '(b h w) e -> b e h w', b=b, e=e, h=h, w=w).contiguous()
        # 这里就是根据token找到对应的词嵌入，然后再将其reshape回原来的形状
        # z_q shape is (N*T, embed_dim, h(H/4), w(W/4))

        # Reshape to original
        z = z.reshape(*shape[:-3], *z.shape[1:]) # z shape is (N, T, embed_dim, h(H/4), w(W/4))
        z_q = z_q.reshape(*shape[:-3], *z_q.shape[1:]) # z_q shape is (N, T, embed_dim, h(H/4), w(W/4))
        tokens = tokens.reshape(*shape[:-3], -1) # tokens shape is (N, T, H/4*W/4)

        # 三者关系查看md文件
        return TokenizerEncoderOutput(z, z_q, tokens)

    def decode(self, z_q: torch.Tensor, should_postprocess: bool = False) -> torch.Tensor:
        '''
        z_quantized shape is (N, T, embed_dim, h(H/4), w(W/4)) 
        should_postprocess: 是否需要后处理 在collect中传入的是True
        '''
        shape = z_q.shape  # (..., E, h, w) (N, T, embed_dim, h(H/4), w(W/4)) 
        z_q = z_q.view(-1, *shape[-3:]) # z_q 展平为 (N*T, embed_dim, h(H/4), w(W/4))
        z_q = self.post_quant_conv(z_q) # 将量化后的向量通过一个1x1卷积转换回原始的z_channels shape is (N*T, z_channels, h(H/4), w(W/4))
        rec = self.decoder(z_q) # 将量化后的向量解码为重建的图像 shape is (N*T, C, H, W)
        rec = rec.reshape(*shape[:-3], *rec.shape[1:])
        if should_postprocess:
            rec = self.postprocess_output(rec)
        return rec

    @torch.no_grad()
    def encode_decode(self, x: torch.Tensor, should_preprocess: bool = False, should_postprocess: bool = False) -> torch.Tensor:
        '''
        x: burnin_obs shape is (N, T, C, H, W);
        shoud_preprocess: 是否需要预处理 True;
        should_postprocess: 是否需要后处理 True
        '''
        z_q = self.encode(x, should_preprocess).z_quantized # z_quantized shape is (N, T, embed_dim, h(H/4), w(W/4)) 
        return self.decode(z_q, should_postprocess)

    def preprocess_input(self, x: torch.Tensor) -> torch.Tensor:
        """x is supposed to be channels first and in [0, 1]"""
        '''
        看起来输入的x已经不是0～255之间的了，而是0～1之间的
        然后将其转换为[-1, 1]之间的张量
        '''
        return x.mul(2).sub(1)

    def postprocess_output(self, y: torch.Tensor) -> torch.Tensor:
        """y is supposed to be channels first and in [-1, 1]"""
        return y.add(1).div(2)
