from collections import OrderedDict
import cv2
from pathlib import Path
import random
import shutil

import numpy as np
import torch
import torch.nn as nn

from episode import Episode


def configure_optimizer(model, learning_rate, weight_decay, *blacklist_module_names):
    '''
    todo 后续可以试试不用这个是否可以训练
    '''
    """Credits to https://github.com/karpathy/minGPT"""
    # separate out all parameters to those that will and won't experience regularizing weight decay
    decay = set()
    no_decay = set()
    whitelist_weight_modules = (torch.nn.Linear, torch.nn.Conv1d)
    blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
    for mn, m in model.named_modules():
        for pn, p in m.named_parameters():
            fpn = '%s.%s' % (mn, pn) if mn else pn  # full param name
            if any([fpn.startswith(module_name) for module_name in blacklist_module_names]):
                no_decay.add(fpn)
            elif 'bias' in pn:
                # all biases will not be decayed
                no_decay.add(fpn)
            elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                # weights of whitelist modules will be weight decayed
                decay.add(fpn)
            elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                # weights of blacklist modules will NOT be weight decayed
                no_decay.add(fpn)

    # validate that we considered every parameter
    param_dict = {pn: p for pn, p in model.named_parameters()}
    inter_params = decay & no_decay
    union_params = decay | no_decay
    # 这里是为了确保没有参数同时出现在 decay 和 no_decay 中
    assert len(inter_params) == 0, f"parameters {str(inter_params)} made it into both decay/no_decay sets!"
    assert len(param_dict.keys() - union_params) == 0, f"parameters {str(param_dict.keys() - union_params)} were not separated into either decay/no_decay set!"

    # create the pytorch optimizer object
    # 这里将 decay 和 no_decay 的参数分开，分别设置不同的 weight_decay
    optim_groups = [
        {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": weight_decay},
        {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
    ]
    optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate)
    return optimizer


def init_weights(module):
    if isinstance(module, (nn.Linear, nn.Embedding)):
        module.weight.data.normal_(mean=0.0, std=0.02)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()
    elif isinstance(module, nn.LayerNorm):
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)


def extract_state_dict(state_dict, module_name):
    '''
    根据给定的模块名称提取状态字典中的参数
    state_dict: 模型的状态字典
    module_name: 模块的名称，通常是模型的前缀
    返回一个新的有序字典，其中只包含指定模块的参数
    这样做的作用就是将存储在一起的模型参数分离出来，便于加载或保存特定模块的参数
    '''
    return OrderedDict({k.split('.', 1)[1]: v for k, v in state_dict.items() if k.startswith(module_name)})


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)


def remove_dir(path, should_ask=False):
    assert path.is_dir()
    if (not should_ask) or input(f"Remove directory : {path} ? [Y/n] ").lower() != 'n':
        shutil.rmtree(path)


def compute_lambda_returns(rewards, values, ends, gamma, lambda_):
    assert rewards.ndim == 2 or (rewards.ndim == 3 and rewards.size(2) == 1)
    assert rewards.shape == ends.shape == values.shape, f"{rewards.shape}, {values.shape}, {ends.shape}"  # (B, T, 1)
    t = rewards.size(1)
    lambda_returns = torch.empty_like(values)
    lambda_returns[:, -1] = values[:, -1]
    lambda_returns[:, :-1] = rewards[:, :-1] + ends[:, :-1].logical_not() * gamma * (1 - lambda_) * values[:, 1:]

    last = values[:, -1]
    for i in list(range(t - 1))[::-1]:
        lambda_returns[:, i] += ends[:, i].logical_not() * gamma * lambda_ * last
        last = lambda_returns[:, i]

    return lambda_returns


class LossWithIntermediateLosses:
    def __init__(self, **kwargs):
        self.loss_total = sum(kwargs.values())
        self.intermediate_losses = {k: v.item() for k, v in kwargs.items()}

    def __truediv__(self, value):
        for k, v in self.intermediate_losses.items():
            self.intermediate_losses[k] = v / value
        self.loss_total = self.loss_total / value
        return self


class EpisodeDirManager:
    '''
    看起来是用来管理训练过程中生成的 episode 的目录，todo 作用
    '''
    def __init__(self, episode_dir: Path, max_num_episodes: int) -> None:
        '''
        episode_dir: 保存 episode 的目录
        max_num_episodes: 最多保存的 episode 数量，如果超过了则删除最旧的一个
        '''
        self.episode_dir = episode_dir
        self.episode_dir.mkdir(parents=False, exist_ok=True)
        self.max_num_episodes = max_num_episodes
        self.best_return = float('-inf') # 最佳 episode 的 return回报，用于判断保存最好回报的Episode

    def save(self, episode: Episode, episode_id: int, epoch: int) -> None:
        if self.max_num_episodes is not None and self.max_num_episodes > 0:
            self._save(episode, episode_id, epoch)

    def _save(self, episode: Episode, episode_id: int, epoch: int) -> None:
        '''
        episode: 需要保存的 Episode 对象，里面包含多个收集的观察数据，包含多伦的数据
        episode_id: 该 episode 的 ID
        epoch: 该 episode 的训练轮数
        '''
        # 检查当前目录下的 episode 数量，如果超过了最大数量，则删除最旧的一个
        ep_paths = [p for p in self.episode_dir.iterdir() if p.stem.startswith('episode_')]
        assert len(ep_paths) <= self.max_num_episodes
        if len(ep_paths) == self.max_num_episodes:
            to_remove = min(ep_paths, key=lambda ep_path: int(ep_path.stem.split('_')[1]))
            to_remove.unlink()
        # 保存当前 episode
        episode.save(self.episode_dir / f'episode_{episode_id}_epoch_{epoch}.pt')

        ep_return = episode.compute_metrics().episode_return
        if ep_return > self.best_return:
            # 如果当前 episode 的回报超过了最佳回报，则保存为最佳 episode
            self.best_return = ep_return
            # 删除之前保存的最佳 episode
            # 这里的逻辑是，如果之前已经保存过最佳 episode，则删除之前的
            # 否则就不删除，因为第一次保存最佳 episode 时没有之前的
            path_best_ep = [p for p in self.episode_dir.iterdir() if p.stem.startswith('best_')]
            assert len(path_best_ep) in (0, 1)
            if len(path_best_ep) == 1:
                path_best_ep[0].unlink()
            # 保存当前 episode 为最佳 episode
            episode.save(self.episode_dir / f'best_episode_{episode_id}_epoch_{epoch}.pt')


class RandomHeuristic:
    '''
    随机动作采样器
    '''
    def __init__(self, num_actions):
        self.num_actions = num_actions

    def act(self, obs):
        assert obs.ndim == 4  # (N, H, W, C)
        n = obs.size(0)
        return torch.randint(low=0, high=self.num_actions, size=(n,))


def make_video(fname, fps, frames):
    assert frames.ndim == 4 # (t, h, w, c)
    t, h, w, c = frames.shape
    assert c == 3

    video = cv2.VideoWriter(str(fname), cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
    for frame in frames:
        video.write(frame[:, :, ::-1])
    video.release()
