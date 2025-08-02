import math
from typing import List

import torch
import torch.nn as nn


class Slicer(nn.Module):
    def __init__(self, max_blocks: int, block_mask: torch.Tensor) -> None:
        '''
        todo 后续调试看看
        max_blocks: 快数量 todo 这个块数量是什么？
        block_mask: torch.Tensor, 掩码，表示哪些位置的tokens需要被保留（主要分为动作和观察两种）
        '''
        super().__init__()
        self.block_size = block_mask.size(0) # 配置中tokens_per_block的大小，如果没猜错也就是tokens中T的大小，也就是时间步的长度
        self.num_kept_tokens = block_mask.sum().long().item() # 每个块的有效信息位置数量，即掩码为1的位置数量
        # torch.where(block_mask)[0] 这行代码获取的是 block_mask 张量中所有值为 True（或非零值）的元素的索引。
        kept_indices = torch.where(block_mask)[0].repeat(max_blocks) 
        offsets = torch.arange(max_blocks).repeat_interleave(self.num_kept_tokens)
        # indices已经是一个全局的索引数组，包含了所有块中需要保留的提取的索引位置
        self.register_buffer('indices', kept_indices + block_mask.size(0) * offsets)

    def compute_slice(self, num_steps: int, prev_steps: int = 0) -> torch.Tensor:
        '''
        num_steps: T(H/4*W/4+1)，tokens的维度
        prev_steps: 在训练world_model时传入的是0
        '''
        total_steps = num_steps + prev_steps # 计算总步数，总的时间步数
        num_blocks = math.ceil(total_steps / self.block_size) # 根据块大小计算能否切分的块数，需要多少个完整的块来容纳这些步数
        indices = self.indices[:num_blocks * self.num_kept_tokens] # 在初始化时预计算的全局索引数组，包含了所有块中需要保留的 token 位置，这里截取前 num_blocks * self.num_kept_tokens 个索引
        # 筛选出在当前时间窗口 [prev_steps, total_steps) 内的索引
        # 减去 prev_steps 将绝对索引转换为相对索引
        return indices[torch.logical_and(prev_steps <= indices, indices < total_steps)] - prev_steps

    def forward(self, *args, **kwargs):
        raise NotImplementedError


class Head(Slicer):
    def __init__(self, max_blocks: int, block_mask: torch.Tensor, head_module: nn.Module) -> None:
        super().__init__(max_blocks, block_mask)
        assert isinstance(head_module, nn.Module)
        self.head_module = head_module

    def forward(self, x: torch.Tensor, num_steps: int, prev_steps: int) -> torch.Tensor:
        '''
        x shape (N, T(H/4*W/4+1), embed_dim) 包含了观察和动作的信息进一步提取的特征
        num_steps: T(H/4*W/4+1)，tokens的维度
        prev_steps: 在训练world_model时传入的是0
        '''
        # self.compute_slice(num_steps, prev_steps)会根据block_mask中掩码不为0的位置计算出需要保留的索引位置
        x_sliced = x[:, self.compute_slice(num_steps, prev_steps)] 
        # 在head_observations中，x_sliced shape is (N, T(H/4*W/4+1 - 1), embed_dim)，因为去除了最后一个观察token
        # 在head_rewards中，x_sliced shape is (N, T(1), embed_dim），因为去仅保留最后一个动作token
        # 在head_ends中，x_sliced shape is (N, T(1), embed_dim)，因为去仅保留最后一个动作token
        return self.head_module(x_sliced)
        # 在head_observations中，x_sliced shape is (N, T(H/4*W/4+1 - 1), obs_vocab_size)
        # 在head_rewards中，x_sliced shape is (N, T(1), 3)，因为去仅保留最后一个动作token
        # 在head_ends中，x_sliced shape is (N, T(1), 2)，因为去仅保留最后一个动作token


class Embedder(nn.Module):
    def __init__(self, max_blocks: int, block_masks: List[torch.Tensor], embedding_tables: List[nn.Embedding]) -> None:
        '''
        max_blocks: 快数量
        block_masks: 动作的掩码，除最后一个位置外，其余位置为0；观察的掩码除最后一个位置外，其余位置为1
        embedding_tables: 嵌入表，包含动作和观察的嵌入表，每个嵌入表的维度为embed_dim，输入的是已经进行了特征提取后的动作和观察
        '''
        super().__init__()
        assert len(block_masks) == len(embedding_tables)
        assert (sum(block_masks) == 1).all()  # block mask are a partition of a block
        self.embedding_dim = embedding_tables[0].embedding_dim # 获取动作特征嵌入embedding
        assert all([e.embedding_dim == self.embedding_dim for e in embedding_tables])
        self.embedding_tables = embedding_tables
        # Slicer是用来切分出tokens中包含的动作和观察信息的模块
        self.slicers = [Slicer(max_blocks, block_mask) for block_mask in block_masks]

    def forward(self, tokens: torch.Tensor, num_steps: int, prev_steps: int) -> torch.Tensor:
        '''
        tokens: (N, T(H/4*W/4+1))， 包含观察和动作的tokens
        num_steps: T(H/4*W/4+1)，tokens的维度
        prev_steps: 在训练world_model时传入的是0

        return 这里返回的就是将tokens中观察和动作分别转换为潜入的矩阵，shape 为 (N, T(H/4*W/4+1), embed_dim)
        '''
        assert tokens.ndim == 2  # x is (B, T)
        output = torch.zeros(*tokens.size(), self.embedding_dim, device=tokens.device) # 创建一个全零的张量，大小为 (N, T(H/4*W/4+1), embed_dim)，用来存储嵌入结果
        for slicer, emb in zip(self.slicers, self.embedding_tables):
            s = slicer.compute_slice(num_steps, prev_steps) # 分别计算出tokens中观察的索引位置和动作的位置的索引
            # tokens[:, s]获取对应观察的引用和动作的引用
            # emb(tokens[:, s])获取对应观察和动作的嵌入，两者因为是不同的信息所以需要不同的嵌入
            output[:, s] = emb(tokens[:, s])
        # 返回嵌入后的结果
        return output
