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
        self.block_size = block_mask.size(0) # 配置中tokens_per_block的大小
        self.num_kept_tokens = block_mask.sum().long().item() # 每个块的有效信息位置，即掩码为1的位置数量
        # torch.where(block_mask)[0] 这行代码获取的是 block_mask 张量中所有值为 True（或非零值）的元素的索引。
        kept_indices = torch.where(block_mask)[0].repeat(max_blocks) 
        offsets = torch.arange(max_blocks).repeat_interleave(self.num_kept_tokens)
        self.register_buffer('indices', kept_indices + block_mask.size(0) * offsets)

    def compute_slice(self, num_steps: int, prev_steps: int = 0) -> torch.Tensor:
        total_steps = num_steps + prev_steps
        num_blocks = math.ceil(total_steps / self.block_size)
        indices = self.indices[:num_blocks * self.num_kept_tokens]
        return indices[torch.logical_and(prev_steps <= indices, indices < total_steps)] - prev_steps

    def forward(self, *args, **kwargs):
        raise NotImplementedError


class Head(Slicer):
    def __init__(self, max_blocks: int, block_mask: torch.Tensor, head_module: nn.Module) -> None:
        super().__init__(max_blocks, block_mask)
        assert isinstance(head_module, nn.Module)
        self.head_module = head_module

    def forward(self, x: torch.Tensor, num_steps: int, prev_steps: int) -> torch.Tensor:
        x_sliced = x[:, self.compute_slice(num_steps, prev_steps)]  # x is (B, T, E)
        return self.head_module(x_sliced)


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
        self.slicers = [Slicer(max_blocks, block_mask) for block_mask in block_masks]

    def forward(self, tokens: torch.Tensor, num_steps: int, prev_steps: int) -> torch.Tensor:
        assert tokens.ndim == 2  # x is (B, T)
        output = torch.zeros(*tokens.size(), self.embedding_dim, device=tokens.device)
        for slicer, emb in zip(self.slicers, self.embedding_tables):
            s = slicer.compute_slice(num_steps, prev_steps)
            output[:, s] = emb(tokens[:, s])
        return output
