from dataclasses import dataclass
from typing import Any, Optional, Union
import sys

from einops import rearrange
import numpy as np
import torch
from torch.distributions.categorical import Categorical
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from dataset import Batch
from envs.world_model_env import WorldModelEnv
from models.tokenizer import Tokenizer
from models.world_model import WorldModel
from utils import compute_lambda_returns, LossWithIntermediateLosses


@dataclass
class ActorCriticOutput:
    logits_actions: torch.FloatTensor # 预测的动作预测 logits_actions shape is (N, 1, act_vocab_size)
    means_values: torch.FloatTensor # 预测的评价均值 means_values shape is (N, 1, 1)


@dataclass
class ImagineOutput:
    observations: torch.ByteTensor
    actions: torch.LongTensor
    logits_actions: torch.FloatTensor
    values: torch.FloatTensor
    rewards: torch.FloatTensor
    ends: torch.BoolTensor


class ActorCritic(nn.Module):
    def __init__(self, act_vocab_size, use_original_obs: bool = False) -> None:
        '''
        act_vocab_size: int, 动作的特征维度
        use_original_obs: bool, 是否使用原始观察数据，默认为False,对应配置文件actor_critic.yaml
        '''
        super().__init__()
        self.use_original_obs = use_original_obs
        self.conv1 = nn.Conv2d(3, 32, 3, stride=1, padding=1)
        self.maxp1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 32, 3, stride=1, padding=1)
        self.maxp2 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(32, 64, 3, stride=1, padding=1)
        self.maxp3 = nn.MaxPool2d(2, 2)
        self.conv4 = nn.Conv2d(64, 64, 3, stride=1, padding=1)
        self.maxp4 = nn.MaxPool2d(2, 2)

        self.lstm_dim = 512
        self.lstm = nn.LSTMCell(1024, self.lstm_dim)
        self.hx, self.cx = None, None

        # 看起来是共享同一个特征，然后分别计算动作和价值
        self.critic_linear = nn.Linear(512, 1)
        self.actor_linear = nn.Linear(512, act_vocab_size)

    def __repr__(self) -> str:
        return "actor_critic"

    def clear(self) -> None:
        self.hx, self.cx = None, None

    def reset(self, n: int, burnin_observations: Optional[torch.Tensor] = None, mask_padding: Optional[torch.Tensor] = None) -> None:
        '''
        n: 环境的数量
        bernin_observations: 重建后的环境观察obs shape (N, T, C, H, W)
        mask_padding: 对于采样的观察不足长度的掩码位 shape （N, T,)

        reset 并不返回任何预测结果，那么reset应该仅仅只是用来重置
        ActorCritic 的状态位 hx 和 cx
        '''
        
        device = self.conv1.weight.device
        # 以下两个应该是用在lstm中的状态位,初始为0
        self.hx = torch.zeros(n, self.lstm_dim, device=device) # (N, lstm_dim)
        self.cx = torch.zeros(n, self.lstm_dim, device=device) # (N, lstm_dim)
        if burnin_observations is not None:
            # burnin_observations.ndim == 5 代表此时观察必须是 (N, T, C, H, W)
            # burnin_observations.size(0) == n 代表N 和环境的数量需要一致，也就是每个环境采样一次
            # mask_padding is not None 就算没有填充也要传入对应的值，即0
            # burnin_observations.shape[:2] == mask_padding.shape 代表 两者的前两个维度都是NT，mask_padding可能后续会自动扩散进行两者合并
            assert burnin_observations.ndim == 5 and burnin_observations.size(0) == n and mask_padding is not None and burnin_observations.shape[:2] == mask_padding.shape
            for i in range(burnin_observations.size(1)): # 遍历每个时间步上的数据
                if mask_padding[:, i].any(): # 这里表示如果mask_padding有任何时间步有有效数据，都要进行一次编码；如果都没有那么久不要进行编码了
                    with torch.no_grad():
                        # 对每一个时间步的环境观察和掩码填充传入给自身
                        self(burnin_observations[:, i], mask_padding[:, i])

    def prune(self, mask: np.ndarray) -> None:
        self.hx = self.hx[mask]
        self.cx = self.cx[mask]

    def forward(self, inputs: torch.FloatTensor, mask_padding: Optional[torch.BoolTensor] = None) -> ActorCriticOutput:
        '''
        inputs: shape is (N, C, H, W)
        mask_padding: shape is (N,)
        '''
        
        assert inputs.ndim == 4 and inputs.shape[1:] == (3, 64, 64) # 判断维度是否是 N C H W, 并且H和W都是64
        assert 0 <= inputs.min() <= 1 and 0 <= inputs.max() <= 1 # 确认是否在[0, 1]之间
        # 要么不传mask_padding，要么mask_padding.ndim == 1，且mask_padding.size(0) == inputs.size(0)，且mask_padding至少有一个True（表明至少有一个N是有包含有效的观察数据）
        assert mask_padding is None or (mask_padding.ndim == 1 and mask_padding.size(0) == inputs.size(0) and mask_padding.any())
        # 将存在有效数据的观察选择出来
        x = inputs[mask_padding] if mask_padding is not None else inputs

        x = x.mul(2).sub(1)  # 将输入数据从[0, 1]范围转换到[-1, 1]范围
        x = F.relu(self.maxp1(self.conv1(x))) # （N, C, H, W） -> （N, C, H/2, W/2）
        x = F.relu(self.maxp2(self.conv2(x))) # （N, C, H/2, W/2） -> （N, C, H/4, W/4）
        x = F.relu(self.maxp3(self.conv3(x))) # （N, C, H/4, W/4） -> （N, C, H/8, W/8）
        x = F.relu(self.maxp4(self.conv4(x))) # （N, C, H/8, W/8） -> （N, C, H/16, W/16）
        x = torch.flatten(x, start_dim=1) # 将特征展平，变为（N, C*H/16*W/16）

        if mask_padding is None:
            # 如果传入None， 则代表没有mask_padding，直接使用全部数据
            self.hx, self.cx = self.lstm(x, (self.hx, self.cx))
        else:
            # 如果有mask_padding，则只对mask_padding为True的部分进行LSTM计算
            # 通过(self.hx[mask_padding], self.cx[mask_padding])提取有效数据进行计算
            self.hx[mask_padding], self.cx[mask_padding] = self.lstm(x, (self.hx[mask_padding], self.cx[mask_padding]))
        
        # self.hx shape is (N, lstm_dim)
        # self.cx shape is (N, lstm_dim)
        '''
        self.hx - 隐藏状态 (hidden state)

        包含 LSTM 单元当前时间步的输出
        作为当前时刻的特征表示，直接用于后续的动作和价值预测
        self.cx - 单元状态 (cell state)

        LSTM 的内部记忆，负责长期信息的传递
        通过门控机制来选择性地保留或忘记信息
        '''

        # self.actor_linear(self.hx) shape is (N, act_vocab_size)
        # self.critic_linear(self.hx) shape is (N, 1)
        # logits_actions shape is (N, 1, act_vocab_size)
        # means_values shape is (N, 1, 1)
        logits_actions = rearrange(self.actor_linear(self.hx), 'b a -> b 1 a')
        means_values = rearrange(self.critic_linear(self.hx), 'b 1 -> b 1 1')

        return ActorCriticOutput(logits_actions, means_values)

    def compute_loss(self, batch: Batch, tokenizer: Tokenizer, world_model: WorldModel, imagine_horizon: int, gamma: float, lambda_: float, entropy_weight: float, **kwargs: Any) -> LossWithIntermediateLosses:
        '''
        batch:
        observations shape (batch_num_samples, 1 + burn_in, channels, height, width)
        actions shape (batch_num_samples, 1 + burn_in, action_dim)
        rewards shape (batch_num_samples, 1 + burn_in)
        dones shape (batch_num_samples, 1 + burn_in)
        imagine_horizon: int, 想象的时间步数，对应配置文件中world_model中的max_block
        '''
        
        assert not self.use_original_obs
        outputs = self.imagine(batch, tokenizer, world_model, horizon=imagine_horizon)

        with torch.no_grad():
            lambda_returns = compute_lambda_returns(
                rewards=outputs.rewards,
                values=outputs.values,
                ends=outputs.ends,
                gamma=gamma,
                lambda_=lambda_,
            )[:, :-1]

        values = outputs.values[:, :-1]

        d = Categorical(logits=outputs.logits_actions[:, :-1])
        log_probs = d.log_prob(outputs.actions[:, :-1])
        loss_actions = -1 * (log_probs * (lambda_returns - values.detach())).mean()
        loss_entropy = - entropy_weight * d.entropy().mean()
        loss_values = F.mse_loss(values, lambda_returns)

        return LossWithIntermediateLosses(loss_actions=loss_actions, loss_values=loss_values, loss_entropy=loss_entropy)

    def imagine(self, batch: Batch, tokenizer: Tokenizer, world_model: WorldModel, horizon: int, show_pbar: bool = False) -> ImagineOutput:
        '''
        batch:
        observations shape (batch_num_samples, 1 + burn_in, channels, height, width)
        actions shape (batch_num_samples, 1 + burn_in, action_dim)
        rewards shape (batch_num_samples, 1 + burn_in)
        dones shape (batch_num_samples, 1 + burn_in)
        '''
        assert not self.use_original_obs
        initial_observations = batch['observations']
        mask_padding = batch['mask_padding']
        assert initial_observations.ndim == 5 and initial_observations.shape[2:] == (3, 64, 64)
        assert mask_padding[:, -1].all() # 至少要有有效数据
        device = initial_observations.device
        wm_env = WorldModelEnv(tokenizer, world_model, device)

        all_actions = []
        all_logits_actions = []
        all_values = []
        all_rewards = []
        all_ends = []
        all_observations = []

        # burnin_observations shape is (B, T-1, C, H, W)
        burnin_observations = torch.clamp(tokenizer.encode_decode(initial_observations[:, :-1], should_preprocess=True, should_postprocess=True), 0, 1) if initial_observations.size(1) > 1 else None
        self.reset(n=initial_observations.size(0), burnin_observations=burnin_observations, mask_padding=mask_padding[:, :-1])

        # initial_observations[:, -1] # shape is (B, C, H, W), 代表最后一个时间步的观察数据
        obs = wm_env.reset_from_initial_observations(initial_observations[:, -1]) # 返回重建后的观察数据，shape is (B, C, H, W)
        for k in tqdm(range(horizon), disable=not show_pbar, desc='Imagination', file=sys.stdout):

            all_observations.append(obs)

            outputs_ac = self(obs)
            action_token = Categorical(logits=outputs_ac.logits_actions).sample() # 对预测的动作进行采样
            obs, reward, done, _ = wm_env.step(action_token, should_predict_next_obs=(k < horizon - 1)) # 执行动作

            all_actions.append(action_token)
            all_logits_actions.append(outputs_ac.logits_actions)
            all_values.append(outputs_ac.means_values)
            all_rewards.append(torch.tensor(reward).reshape(-1, 1))
            all_ends.append(torch.tensor(done).reshape(-1, 1))

        self.clear()

        return ImagineOutput(
            observations=torch.stack(all_observations, dim=1).mul(255).byte(),      # (B, T, C, H, W) in [0, 255]
            actions=torch.cat(all_actions, dim=1),                                  # (B, T)
            logits_actions=torch.cat(all_logits_actions, dim=1),                    # (B, T, #actions)
            values=rearrange(torch.cat(all_values, dim=1), 'b t 1 -> b t'),         # (B, T)
            rewards=torch.cat(all_rewards, dim=1).to(device),                       # (B, T)
            ends=torch.cat(all_ends, dim=1).to(device),                             # (B, T)
        )
