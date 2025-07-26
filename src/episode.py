from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path

import torch


@dataclass
class EpisodeMetrics:
    episode_length: int # 生命周期的长度，也就是收集到的观察数据的数量
    episode_return: float # 生命周期的回报，也就是收集到的奖励的总和


@dataclass
class Episode:
    observations: torch.ByteTensor
    actions: torch.LongTensor
    rewards: torch.FloatTensor
    ends: torch.LongTensor
    mask_padding: torch.BoolTensor

    def __post_init__(self):
        assert len(self.observations) == len(self.actions) == len(self.rewards) == len(self.ends) == len(self.mask_padding)
        if self.ends.sum() > 0:
            idx_end = torch.argmax(self.ends) + 1
            self.observations = self.observations[:idx_end]
            self.actions = self.actions[:idx_end]
            self.rewards = self.rewards[:idx_end]
            self.ends = self.ends[:idx_end]
            self.mask_padding = self.mask_padding[:idx_end]

    def __len__(self) -> int:
        return self.observations.size(0)

    def merge(self, other: Episode) -> Episode:
        return Episode(
            torch.cat((self.observations, other.observations), dim=0),
            torch.cat((self.actions, other.actions), dim=0),
            torch.cat((self.rewards, other.rewards), dim=0),
            torch.cat((self.ends, other.ends), dim=0),
            torch.cat((self.mask_padding, other.mask_padding), dim=0),
        )

    def segment(self, start: int, stop: int, should_pad: bool = False) -> Episode:
        '''
        start: int, 片段的起始索引
        stop: int, 片段的结束索引，传入的是本身的长度；
        should_pad: bool, 是否需要填充片段到指定长度，默认为False，如果已经收集的episode长度小于stop，则会填充0，使用时传入的是True
        '''
        assert start < len(self) and stop > 0 and start < stop
        # 以下操作就是为了判断是否有足够的数据量来满足片段的长度要求
        padding_length_right = max(0, stop - len(self)) 
        padding_length_left = max(0, -start)
        assert padding_length_right == padding_length_left == 0 or should_pad # 判断是否开启填充，如果不需要填充则声明周期的长度要有足够的数据量

        def pad(x):
            # 构建一个形如[0, 0, 0, 0, padding_length_right]填充张量
            pad_right = torch.nn.functional.pad(x, [0 for _ in range(2 * x.ndim - 1)] + [padding_length_right]) if padding_length_right > 0 else x
            # [0, 0, 0, 0, padding_length_left, 0]
            # 注意，对应pad来说，传入的input指定的维度是从后往前，从左往右排列的
            # 所以这里的0, 0, 0, 0, padding_length_right]和[0, 0, 0, 0, padding_length_left, 0]都是时间维度
            return torch.nn.functional.pad(pad_right, [0 for _ in range(2 * x.ndim - 2)] + [padding_length_left, 0]) if padding_length_left > 0 else pad_right

        # 确保 start 和 stop 在有效范围内
        start = max(0, start)
        stop = min(len(self), stop)
        # 截取片段
        segment = Episode(
            self.observations[start:stop],
            self.actions[start:stop],
            self.rewards[start:stop],
            self.ends[start:stop],
            self.mask_padding[start:stop],
        )

        # 填充数据
        segment.observations = pad(segment.observations)
        segment.actions = pad(segment.actions)
        segment.rewards = pad(segment.rewards)
        segment.ends = pad(segment.ends)
        # todo mask_padding在哪里设置为True的？
        segment.mask_padding = torch.cat((torch.zeros(padding_length_left, dtype=torch.bool), segment.mask_padding, torch.zeros(padding_length_right, dtype=torch.bool)), dim=0)

        return segment

    def compute_metrics(self) -> EpisodeMetrics:
        return EpisodeMetrics(len(self), self.rewards.sum())

    def save(self, path: Path) -> None:
        torch.save(self.__dict__, path)


'''
# 解析 torch.nn.functional.pad 的维度填充控制

`torch.nn.functional.pad` 通过一个特殊的参数列表来控制在哪些维度上进行填充，以及每个维度填充的大小。

## 基本语法

```python
torch.nn.functional.pad(input, pad, mode='constant', value=0)
```

其中最关键的是 `pad` 参数，它是一个列表或元组，按照特定的顺序指定每个维度两侧的填充量。

## pad 参数格式

`pad` 的格式为：
```
[pad_left_dim_n, pad_right_dim_n, ..., pad_left_dim_1, pad_right_dim_1]
```

重要特点：
1. 填充是**从最后一个维度开始**向前指定的
2. 每个维度需要两个值，分别表示该维度左侧和右侧的填充量
3. 不必为所有维度指定填充，未指定的维度默认不填充

## 在 Episode 类中的应用

```python
pad_right = torch.nn.functional.pad(x, [0 for _ in range(2 * x.ndim - 1)] + [padding_length_right])
```

这行代码生成的填充参数解析：

1. 对于一个 `[T, C, H, W]` 形状的张量 (ndim = 4)：
   - 生成 `[0, 0, 0, 0, 0, 0, 0, padding_length_right]` 的填充列表
   - 这表示只在第一个维度（时间维度）的右侧填充 `padding_length_right` 个元素

2. 填充参数的含义：
   - `[0, 0]`：W 维度不填充（宽度）
   - `[0, 0]`：H 维度不填充（高度）
   - `[0, 0]`：C 维度不填充（通道）
   - `[0, padding_length_right]`：T 维度（时间）右侧填充

同理，左侧填充的代码：
```python
torch.nn.functional.pad(pad_right, [0 for _ in range(2 * x.ndim - 2)] + [padding_length_left, 0])
```

生成类似的填充参数，但只在第一个维度（时间维度）的左侧填充。

## 为什么这样设计

代码使用了一个巧妙的技巧来构建填充参数：
- `[0 for _ in range(2 * x.ndim - 1)]` 创建了一串0，对应除了第一个维度右侧外的所有填充位置
- `+ [padding_length_right]` 在列表末尾添加了第一个维度右侧的填充量

这种设计确保了只在时间维度上进行填充，保持了观察数据中的特征维度不变，这对于保留数据的语义一致性至关重要。

'''