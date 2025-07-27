"""
Credits to https://github.com/karpathy/minGPT
"""

from dataclasses import dataclass
import math
from typing import Optional

from einops import rearrange
import torch
import torch.nn as nn
from torch.nn import functional as F

from .kv_caching import KeysValues, KVCache


@dataclass
class TransformerConfig:
    tokens_per_block: int
    max_blocks: int
    attention: str

    num_layers: int
    num_heads: int
    embed_dim: int

    embed_pdrop: float
    resid_pdrop: float
    attn_pdrop: float

    @property
    def max_tokens(self):
        '''
        这个看起来是获取最长的tokens数量
        '''
        return self.tokens_per_block * self.max_blocks


class Transformer(nn.Module):
    def __init__(self, config: TransformerConfig) -> None:
        '''
         config: 世界模型的配置参数，TransformerConfig类型
            tokens_per_block: 17
            max_blocks: 20
            attention: 'causal'
            num_layers: 10
            num_heads: 4
            embed_dim: 256
            embed_pdrop: 0.1
            resid_pdrop: 0.1
            attn_pdrop: 0.1
        '''
        super().__init__()
        self.config = config
        self.drop = nn.Dropout(config.embed_pdrop)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.num_layers)])
        self.ln_f = nn.LayerNorm(config.embed_dim)

    def generate_empty_keys_values(self, n: int, max_tokens: int) -> KeysValues:
        device = self.ln_f.weight.device  # Assumption that all submodules are on the same device
        return KeysValues(n, self.config.num_heads, max_tokens, self.config.embed_dim, self.config.num_layers, device)

    def forward(self, sequences: torch.Tensor, past_keys_values: Optional[KeysValues] = None) -> torch.Tensor:
        '''
        sequences: shape is (N, T(H/4*W/4+1), embed_dim)，包含了观察token和动作的信息
        past_keys_values: Optional[KeysValues]，如果是训练world_model时传入的是None
        '''
        assert past_keys_values is None or len(past_keys_values) == len(self.blocks)
        x = self.drop(sequences)
        for i, block in enumerate(self.blocks):
            x = block(x, None if past_keys_values is None else past_keys_values[i])

        x = self.ln_f(x)
        return x


class Block(nn.Module):
    def __init__(self, config: TransformerConfig) -> None:
        '''
        config: 世界模型的配置参数，TransformerConfig类型
            tokens_per_block: 17
            max_blocks: 20
            attention: 'causal'
            num_layers: 10
            num_heads: 4
            embed_dim: 256
            embed_pdrop: 0.1
            resid_pdrop: 0.1
            attn_pdrop: 0.1

            每层transformer包含一个自注意力层和一个前馈网络层，在输入时进行了一次层归一化，在输入给mlp层时进行了一次层归一化
        '''
        super().__init__()
        self.ln1 = nn.LayerNorm(config.embed_dim)
        self.ln2 = nn.LayerNorm(config.embed_dim)
        self.attn = SelfAttention(config)
        self.mlp = nn.Sequential(
            nn.Linear(config.embed_dim, 4 * config.embed_dim),
            nn.GELU(),
            nn.Linear(4 * config.embed_dim, config.embed_dim),
            nn.Dropout(config.resid_pdrop),
        )

    def forward(self, x: torch.Tensor, past_keys_values: Optional[KeysValues] = None) -> torch.Tensor:
        '''
        x: shape is (N, T(H/4*W/4+1), embed_dim)，包含了观察token和动作的信息
        past_keys_values: Optional[KeysValues]，如果是训练world_model时传入的是None todo 这里的参数应该是KVCache类型
        '''
        x_attn = self.attn(self.ln1(x), past_keys_values)
        x = x + x_attn
        x = x + self.mlp(self.ln2(x))
        return x


class SelfAttention(nn.Module):
    def __init__(self, config: TransformerConfig) -> None:
        '''
        tokens_per_block: 17
        max_blocks: 20
        attention: 'causal'
        num_layers: 10
        num_heads: 4
        embed_dim: 256
        embed_pdrop: 0.1
        resid_pdrop: 0.1
        attn_pdrop: 0.1
        '''
        super().__init__()
        assert config.embed_dim % config.num_heads == 0
        assert config.attention in ('causal', 'block_causal')
        self.num_heads = config.num_heads # 注意力头的数量
        self.key = nn.Linear(config.embed_dim, config.embed_dim)
        self.query = nn.Linear(config.embed_dim, config.embed_dim)
        self.value = nn.Linear(config.embed_dim, config.embed_dim)
        self.attn_drop = nn.Dropout(config.attn_pdrop)
        self.resid_drop = nn.Dropout(config.resid_pdrop)
        self.proj = nn.Linear(config.embed_dim, config.embed_dim)

        # todo causal_mask这个下三角矩阵的作用？
        causal_mask = torch.tril(torch.ones(config.max_tokens, config.max_tokens))
        # 创建一个块对角矩阵，由max_blocks个tokens_per_block × tokens_per_block的全1矩阵沿对角线排列组成
        # https://blog.csdn.net/wuruoting_claire/article/details/127588377
        block_causal_mask = torch.max(causal_mask, torch.block_diag(*[torch.ones(config.tokens_per_block, config.tokens_per_block) for _ in range(config.max_blocks)]))
        # todo 以上两个矩阵的区别的作用？todo
        self.register_buffer('mask', causal_mask if config.attention == 'causal' else block_causal_mask)

    def forward(self, x: torch.Tensor, kv_cache: Optional[KVCache] = None) -> torch.Tensor:
        '''
        x: shape is (N, T(H/4*W/4+1), embed_dim)，包含了观察token和动作的信息
        kv_cache: 如果是训练world_model时传入的是None
        '''
        B, T, C = x.size() # (N, T(H/4*W/4+1), embed_dim)
        if kv_cache is not None:
            # todo 啥时候传入的不是None
            # 应该是在推理时缓存，用于增量解码，支持生成时的高效推理，避免重复计算历史tokens的K、V
            b, nh, L, c = kv_cache.shape
            assert nh == self.num_heads and b == B and c * nh == C
        else:
            L = 0 # 表示之前缓存的长度

        # self.query(x) shape is (N, T(H/4*W/4+1), embed_dim)
        # view(B, T, num_heads, C // num_heads) shape is (N, T(H/4*W/4+1), num_heads, embed_dim // num_heads)
        # transpose(1, 2) shape is (N, num_heads, T(H/4*W/4+1), embed_dim // num_heads)
        # 最终q, k, v的shape都是 (N, num_heads, T(H/4*W/4+1), embed_dim // num_heads)
        q = self.query(x).view(B, T, self.num_heads, C // self.num_heads).transpose(1, 2) 
        k = self.key(x).view(B, T, self.num_heads, C // self.num_heads).transpose(1, 2)    
        v = self.value(x).view(B, T, self.num_heads, C // self.num_heads).transpose(1, 2)   

        if kv_cache is not None:
            # todo 啥时候不是None
            # 训练时跳过（kv_cache=None）
            # 推理时将新的K、V添加到缓存中，获取完整的K、V序列
            kv_cache.update(k, v)
            k, v = kv_cache.get()

        # k.transpose(-2, -1) shape is (N, num_heads, embed_dim // num_heads, T(H/4*W/4+1))
        # q @ k.transpose(-2, -1) shape is (N, num_heads, T(H/4*W/4+1), T(H/4*W/4+1))
        # 计算每个query与所有key的相似度分数
        # 缩放因子：1/√d_k 防止点积值过大，确保softmax输出平滑
        # att[b,h,i,j] 表示第b个样本、第h个头中，位置i对位置j的注意力原始分数
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        '''
        原理：

        Causal Mask：下三角矩阵，确保位置i只能看到位置≤i的信息
        Block Causal Mask：块对角结构，允许块内全连接，块间因果
        作用：

        防止信息泄露（未来信息不能影响当前决策）
        在IRIS中确保世界模型的时序一致性
        todo 重新多看看transformer的代码
        '''
        att = att.masked_fill(self.mask[L:L + T, :L + T] == 0, float('-inf'))
        # Softmax：将注意力分数转换为概率分布
        # 数学意义：att[b,h,i,j] 现在表示位置i对位置j的注意力权重，每行和为1
        att = F.softmax(att, dim=-1)
        # Dropout：训练时随机置零部分注意力连接，防止过拟合
        att = self.attn_drop(att)
        # att shape is (N, num_heads, T(H/4*W/4+1), T(H/4*W/4+1))
        # v shape is (N, num_heads, T(H/4*W/4+1), embed_dim // num_heads)
        # 最终y shape is (N, T(H/4*W/4+1), embed_dim)
        # 计算注意力输出：每个位置的表示是所有位置的值的加权平均
        # att @ v shape is (N, num_heads, T(H/4*W/4+1), embed_dim // num_heads)
        # 将注意力权重应用于值向量，得到每个位置的加权表示
        y = att @ v
        # y shape is (N, T(H/4*W/4+1), embed_dim)
        # 将多个注意力头的输出拼接回原始嵌入维度 作用：整合不同头学到的特征表示
        y = rearrange(y, 'b h t e -> b t (h e)')

        '''
        原理：

        线性投影：将拼接后的特征映射回输出空间
        Dropout：正则化防止过拟合
        意义：

        允许模型学习如何组合多头注意力的信息
        为后续的残差连接准备输出
        '''
        y = self.resid_drop(self.proj(y))

        return y
