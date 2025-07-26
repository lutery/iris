from collections import deque
import math
from pathlib import Path
import random
from typing import Dict, List, Optional, Tuple

import psutil
import torch

from episode import Episode

Batch = Dict[str, torch.Tensor]


class EpisodesDataset:
    def __init__(self, max_num_episodes: Optional[int] = None, name: Optional[str] = None) -> None:
        '''
        max_num_episodes: None or int, None表示不限制最大episode数量,EpisodesDatasetRamMonitoring子类传入的是None
        name: str, 用于标识数据集的名称，默认为'dataset'
        '''
        self.max_num_episodes = max_num_episodes
        self.name = name if name is not None else 'dataset'
        self.num_seen_episodes = 0 # 统计已经存储了多少个 episodes
        self.episodes = deque() # 以deque的形式存储episodes，方便在添加新episode时删除最旧的episode，每个episode存储着一个
        self.episode_id_to_queue_idx = dict() # 因为如果超过了缓冲区的大小，后续会清理旧的数据，所以需要一个字典来记录每个episode的ID和其在deque中的索引
        # self.newly_modified_episodes 存储最近修改/新增的episode ID
        # self.newly_deleted_episodes 存储最近删除的episode ID
        self.newly_modified_episodes, self.newly_deleted_episodes = set(), set()

    def __len__(self) -> int:
        return len(self.episodes)

    def clear(self) -> None:
        self.episodes = deque()
        self.episode_id_to_queue_idx = dict()

    def add_episode(self, episode: Episode) -> int:
        if self.max_num_episodes is not None and len(self.episodes) == self.max_num_episodes:
            # 如果达到最大episode数量，则删除最旧的episode
            self._popleft()
        episode_id = self._append_new_episode(episode)
        # 返回新添加的episode的ID，可以用这个id找到对应的存储位置
        return episode_id

    def get_episode(self, episode_id: int) -> Episode:
        '''
        根据episode_id获取对应的Episode对象
        episode_id: int, 要获取的episode的ID
        '''
        assert episode_id in self.episode_id_to_queue_idx
        queue_idx = self.episode_id_to_queue_idx[episode_id]
        return self.episodes[queue_idx]

    def update_episode(self, episode_id: int, new_episode: Episode) -> None:
        '''
        episode_id: int, 要更新的episode的ID
        new_episode: Episode, 新的episode数据，将会合并到已有的episode中
        '''
        assert episode_id in self.episode_id_to_queue_idx
        queue_idx = self.episode_id_to_queue_idx[episode_id] # 获取要更新的episode在deque中的索引
        merged_episode = self.episodes[queue_idx].merge(new_episode) # 合并新的episode数据到已有的episode中，从这里来看，对于Episode类来说，merge方法应该是将两个episode合并成一个新的episode，那么由于收集数据导致被截断的episode会被合并成一个完整的episode
        # 会合并一般是发生在退出收集循环后，将剩下的未完成的episode合并进来，其余的时候可能就reset了，重新收集存储
        self.episodes[queue_idx] = merged_episode # 更新deque中的episode
        self.newly_modified_episodes.add(episode_id) # 将更新的episode ID添加到新修改的episode ID集合中，因为是set，所以会自动去重

    def _popleft(self) -> Episode:
        id_to_delete = [k for k, v in self.episode_id_to_queue_idx.items() if v == 0] # 这里是为了找到索引为0的episode ID
        assert len(id_to_delete) == 1 # 确认只有一个episode的索引为0
        self.newly_deleted_episodes.add(id_to_delete[0]) # 将要删除的episode ID添加到新删除的episode ID集合中
        self.episode_id_to_queue_idx = {k: v - 1 for k, v in self.episode_id_to_queue_idx.items() if v > 0} # 因为删除了索引为0的episode，所以需要将其他episode的索引减1
        return self.episodes.popleft() # 删除最旧的episode并返回它

    def _append_new_episode(self, episode):
        '''
        填充一个完整生命周期的episode到数据集中，并返回其ID。
        '''
        episode_id = self.num_seen_episodes
        self.episode_id_to_queue_idx[episode_id] = len(self.episodes) # 将收集的episode的ID和其在队列中的索引对应起来
        self.episodes.append(episode) # 
        self.num_seen_episodes += 1 # 更新已接收episode数据的数量
        self.newly_modified_episodes.add(episode_id) # 存储最新的episode ID
        return episode_id

    def sample_batch(self, batch_num_samples: int, sequence_length: int, sample_from_start: bool = True) -> Batch:
        return self._collate_episodes_segments(self._sample_episodes_segments(batch_num_samples, sequence_length, sample_from_start))

    def _sample_episodes_segments(self, batch_num_samples: int, sequence_length: int, sample_from_start: bool) -> List[Episode]:
        sampled_episodes = random.choices(self.episodes, k=batch_num_samples)
        sampled_episodes_segments = []
        for sampled_episode in sampled_episodes:
            if sample_from_start:
                start = random.randint(0, len(sampled_episode) - 1)
                stop = start + sequence_length
            else:
                stop = random.randint(1, len(sampled_episode))
                start = stop - sequence_length
            sampled_episodes_segments.append(sampled_episode.segment(start, stop, should_pad=True))
            assert len(sampled_episodes_segments[-1]) == sequence_length
        return sampled_episodes_segments

    def _collate_episodes_segments(self, episodes_segments: List[Episode]) -> Batch:
        episodes_segments = [e_s.__dict__ for e_s in episodes_segments]
        batch = {}
        for k in episodes_segments[0]:
            batch[k] = torch.stack([e_s[k] for e_s in episodes_segments])
        batch['observations'] = batch['observations'].float() / 255.0  # int8 to float and scale
        return batch

    def traverse(self, batch_num_samples: int, chunk_size: int):
        for episode in self.episodes:
            chunks = [episode.segment(start=i * chunk_size, stop=(i + 1) * chunk_size, should_pad=True) for i in range(math.ceil(len(episode) / chunk_size))]
            batches = [chunks[i * batch_num_samples: (i + 1) * batch_num_samples] for i in range(math.ceil(len(chunks) / batch_num_samples))]
            for b in batches:
                yield self._collate_episodes_segments(b)

    def update_disk_checkpoint(self, directory: Path) -> None:
        assert directory.is_dir()
        for episode_id in self.newly_modified_episodes:
            episode = self.get_episode(episode_id)
            episode.save(directory / f'{episode_id}.pt')
        for episode_id in self.newly_deleted_episodes:
            (directory / f'{episode_id}.pt').unlink()
        self.newly_modified_episodes, self.newly_deleted_episodes = set(), set()

    def load_disk_checkpoint(self, directory: Path) -> None:
        assert directory.is_dir() and len(self.episodes) == 0
        episode_ids = sorted([int(p.stem) for p in directory.iterdir()])
        self.num_seen_episodes = episode_ids[-1] + 1
        for episode_id in episode_ids:
            episode = Episode(**torch.load(directory / f'{episode_id}.pt'))
            self.episode_id_to_queue_idx[episode_id] = len(self.episodes)
            self.episodes.append(episode)


class EpisodesDatasetRamMonitoring(EpisodesDataset):
    """
    Prevent episode dataset from going out of RAM.
    Warning: % looks at system wide RAM usage while G looks only at process RAM usage.
    """
    def __init__(self, max_ram_usage: str, name: Optional[str] = None) -> None:
        super().__init__(max_num_episodes=None, name=name)
        self.max_ram_usage = max_ram_usage # 这个感觉是限制最大内存使用量
        self.num_steps = 0
        self.max_num_steps = None

        max_ram_usage = str(max_ram_usage)
        # 创建一个检查内存使用量的函数，如果超过了返回False，否则返回True
        if max_ram_usage.endswith('%'):
            m = int(max_ram_usage.split('%')[0])
            assert 0 < m < 100
            self.check_ram_usage = lambda: psutil.virtual_memory().percent > m
        else:
            assert max_ram_usage.endswith('G')
            m = float(max_ram_usage.split('G')[0])
            self.check_ram_usage = lambda: psutil.Process().memory_info()[0] / 2 ** 30 > m

    def clear(self) -> None:
        super().clear()
        self.num_steps = 0

    def add_episode(self, episode: Episode) -> int:
        if self.max_num_steps is None and self.check_ram_usage():
            self.max_num_steps = self.num_steps
        self.num_steps += len(episode)
        while (self.max_num_steps is not None) and (self.num_steps > self.max_num_steps):
            self._popleft()
        episode_id = self._append_new_episode(episode)
        return episode_id

    def _popleft(self) -> Episode:
        episode = super()._popleft()
        self.num_steps -= len(episode)
        return episode
