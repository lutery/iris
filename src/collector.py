import random
import sys
from typing import List, Optional, Union

from einops import rearrange
import numpy as np
import torch
from tqdm import tqdm
import wandb

from agent import Agent
from dataset import EpisodesDataset
from envs import SingleProcessEnv, MultiProcessEnv
from episode import Episode
from utils import EpisodeDirManager, RandomHeuristic


class Collector:
    def __init__(self, env: Union[SingleProcessEnv, MultiProcessEnv], dataset: EpisodesDataset, episode_dir_manager: EpisodeDirManager) -> None:
        '''
        env: 游戏环境实例，SingleProcessEnv或MultiProcessEnv
        dataset: EpisodesDataset实例，用于存储收集到的观察数据
        episode_dir_manager: EpisodeDirManager实例，用于管理episode的保存路径
        该类负责从环境中收集观察数据，并将其存储到dataset中，同时管理episode的保存路径
        '''
        
        self.env = env
        self.dataset = dataset
        self.episode_dir_manager = episode_dir_manager
        self.obs = self.env.reset()
        self.episode_ids = [None] * self.env.num_envs # 存储每一个环境的对应在dataset缓冲区的最新的episode id，可以用这个id来找到对应的Episode实例
        self.heuristic = RandomHeuristic(self.env.num_actions) # 这个看起来是随机动作选择器

    @torch.no_grad()
    def collect(self, agent: Agent, epoch: int, epsilon: float, should_sample: bool, temperature: float, burn_in: int, *, num_steps: Optional[int] = None, num_episodes: Optional[int] = None):
        '''
        agent: Agent实例，包含策略网络\世界模型等
        epoch: 当前训练的轮次
        其余的参数则是配置文件中的参数，即trainer.yaml中collection.train.config
        epsilon: 0.01
        should_sample: True
        temperature: 1.0
        num_steps: 200
        burn_in: ${training.actor_critic.burn_in} 配置文件中是20
        num_episodes：在train中是None，表示不限制收集的episode数量，但是在test中是一个整数，表示收集的episode数量
        '''
        
        assert self.env.num_actions == agent.world_model.act_vocab_size
        assert 0 <= epsilon <= 1

        assert (num_steps is None) != (num_episodes is None)
        # 判断是否应该停止收集数据，条件是达到指定的步数或episode数量
        should_stop = lambda steps, episodes: steps >= num_steps if num_steps is not None else episodes >= num_episodes

        to_log = [] # 存储收集的数据中的指标信息，后续会将这些指标信息记录到wandb中
        steps, episodes = 0, 0 # 存储已经收集的环境步数，存储已经收集的生命周期的数量
        returns = [] # 存储每个episode的回报，也就是收集到的奖励的总和
        # 存储收集到的观察数据，动作，奖励和结束标识
        observations, actions, rewards, dones = [], [], [], []

        burnin_obs_rec, mask_padding = None, None # burnin_obs_rec存储对观察进行编码重建后的结果，mask_padding是padding的mask ｜ mask_padding存储的是从环境中截取后不足的长度补充的填充
        if set(self.episode_ids) != {None} and burn_in > 0:
            # 获取所有环境当前的episode
            current_episodes = [self.dataset.get_episode(episode_id) for episode_id in self.episode_ids]
            # 从每个episode中截取最后burn_in长度的段落
            segmented_episodes = [episode.segment(start=len(episode) - burn_in, stop=len(episode), should_pad=True) for episode in current_episodes]
            # 这里将所有的mask_padding和observations堆叠成一个batch
            mask_padding = torch.stack([episode.mask_padding for episode in segmented_episodes], dim=0).to(agent.device)
            burnin_obs = torch.stack([episode.observations for episode in segmented_episodes], dim=0).float().div(255).to(agent.device)
            # 对环境进行编码重建，得到重建后的观察，burnin_obs shape is (N, T, C, H, W) 
            burnin_obs_rec = torch.clamp(agent.tokenizer.encode_decode(burnin_obs, should_preprocess=True, should_postprocess=True), 0, 1)

        agent.actor_critic.reset(n=self.env.num_envs, burnin_observations=burnin_obs_rec, mask_padding=mask_padding)
        # 有指定步数就按步数收集数据，否则按episode数量收集数据
        pbar = tqdm(total=num_steps if num_steps is not None else num_episodes, desc=f'Experience collection ({self.dataset.name})', file=sys.stdout)

        while not should_stop(steps, episodes):
            
            # self.obs存储当前的obs
            observations.append(self.obs)
            # 在这里将obs转换为0～1之间的浮点数，通道数为3，大小为64x64
            obs = rearrange(torch.FloatTensor(self.obs).div(255), 'n h w c -> n c h w').to(agent.device)
            # act shape is (N, 1)
            act = agent.act(obs, should_sample=should_sample, temperature=temperature).cpu().numpy()

            if random.random() < epsilon:
                # 如果随机数小于epsilon，则使用随机动作采样器
                act = self.heuristic.act(obs).cpu().numpy()

            # 执行动作，的到下一个观察，奖励和结束标识，并存储起来
            # 看起来存储区域是按照当前的观察，当前观察下的动作，奖励和结束标识来存储的
            # 在不断的step中，会自动根据是否结束来重置环境
            self.obs, reward, done, _ = self.env.step(act)

            actions.append(act)
            rewards.append(reward)
            dones.append(done)

            # 获取返回未结束或者新结束的环境数量，用来更新进度条
            # 因为是并行环境，如果一个环境已经结束了，那么执行的step是无效的
            # 属于无效步数，不需要统计更新
            # 如果是新结束的，那么导致结束执行的那一步是有效的，是需要进行统计的
            # 这里是为什么要区分新结束或者已结束的原因
            # 根据以上的逻辑，在收集数据时，一般都是有效的，因为会自动重制环境，所以基本都是新结束或者未结束的
            new_steps = len(self.env.mask_new_dones)
            # 更新进度条
            steps += new_steps
            pbar.update(new_steps if num_steps is not None else 0)

            # Warning: with EpisodicLifeEnv + MultiProcessEnv, reset is ignored if not a real done.
            # Thus, segments of experience following a life loss and preceding a general done are discarded.
            # Not a problem with a SingleProcessEnv.

            if self.env.should_reset():
                # 如果已经结束的环境数量大于等于应该等待的环境数量比例，则重置环境

                # 首先将收集到的观察，动作，奖励和结束标识添加到数据集中
                self.add_experience_to_dataset(observations, actions, rewards, dones)

                new_episodes = self.env.num_envs
                episodes += new_episodes # 更新已经收集的episode数量
                pbar.update(new_episodes if num_episodes is not None else 0)

                for episode_id in self.episode_ids:
                    episode = self.dataset.get_episode(episode_id) # 获取对应id存储的episode数据
                    self.episode_dir_manager.save(episode, episode_id, epoch) # 保存episode数据到指定路径
                    metrics_episode = {k: v for k, v in episode.compute_metrics().__dict__.items()}
                    metrics_episode['episode_num'] = episode_id
                    # 统计该周期内动作的选择倾向，是指向同一个动作还是分布均匀
                    metrics_episode['action_histogram'] = wandb.Histogram(np_histogram=np.histogram(episode.actions.numpy(), bins=np.arange(0, self.env.num_actions + 1) - 0.5, density=True))
                    to_log.append({f'{self.dataset.name}/{k}': v for k, v in metrics_episode.items()})
                    returns.append(metrics_episode['episode_return'])

                self.obs = self.env.reset() # 重置环境，获取新的观察数据
                self.episode_ids = [None] * self.env.num_envs # 重置episode_ids为None，表示当前没有收集到数据
                agent.actor_critic.reset(n=self.env.num_envs) # 重置actor_critic的状态
                observations, actions, rewards, dones = [], [], [], [] # 清空收集到的观察，动作，奖励和结束标识

        # Add incomplete episodes to dataset, and complete them later.
        if len(observations) > 0:
            # 将最后收集到的观察，动作，奖励和结束标识添加到数据集中
            self.add_experience_to_dataset(observations, actions, rewards, dones)

        agent.actor_critic.clear()

        metrics_collect = {
            '#episodes': len(self.dataset),
            '#steps': sum(map(len, self.dataset.episodes)),
        }
        if len(returns) > 0:
            metrics_collect['return'] = np.mean(returns)
        metrics_collect = {f'{self.dataset.name}/{k}': v for k, v in metrics_collect.items()}
        to_log.append(metrics_collect)

        return to_log

    def add_experience_to_dataset(self, observations: List[np.ndarray], actions: List[np.ndarray], rewards: List[np.ndarray], dones: List[np.ndarray]) -> None:
        assert len(observations) == len(actions) == len(rewards) == len(dones)
        # 这里将T N转换为N T的形式，即将时间步和环境数量进行交换,因为在收集数据时都是按照时间的维度填充到 list中
        for i, (o, a, r, d) in enumerate(zip(*map(lambda arr: np.swapaxes(arr, 0, 1), [observations, actions, rewards, dones]))):  # Make everything (N, T, ...) instead of (T, N, ...)
            # 遍历每一个环境的完整时间步的环境观察、动作，奖励和结束标识
            # todo 后续判断这里是存储一个完整的生命周期还是存储着一个完整生命周期以及多个生命周期的片段
            episode = Episode(
                observations=torch.ByteTensor(o).permute(0, 3, 1, 2).contiguous(),  # channel-first
                actions=torch.LongTensor(a),
                rewards=torch.FloatTensor(r),
                ends=torch.LongTensor(d),
                mask_padding=torch.ones(d.shape[0], dtype=torch.bool),
            )
            if self.episode_ids[i] is None:
                # 如果对应的环境没有存储过数据，那么就新增
                self.episode_ids[i] = self.dataset.add_episode(episode)
            else:
                # 如果对应的环境已经存储过数据，那么就更新，将收集到的数据和之前已经存储的数据继续拼接起来
                self.dataset.update_episode(self.episode_ids[i], episode)
