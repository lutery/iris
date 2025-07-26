from pathlib import Path

import torch
from torch.distributions.categorical import Categorical
import torch.nn as nn

from models.actor_critic import ActorCritic
from models.tokenizer import Tokenizer
from models.world_model import WorldModel
from utils import extract_state_dict


class Agent(nn.Module):
    def __init__(self, tokenizer: Tokenizer, world_model: WorldModel, actor_critic: ActorCritic):
        super().__init__()
        self.tokenizer = tokenizer
        self.world_model = world_model
        self.actor_critic = actor_critic

    @property
    def device(self):
        return self.actor_critic.conv1.weight.device

    def load(self, path_to_checkpoint: Path, device: torch.device, load_tokenizer: bool = True, load_world_model: bool = True, load_actor_critic: bool = True) -> None:
        agent_state_dict = torch.load(path_to_checkpoint, map_location=device)
        if load_tokenizer:
            self.tokenizer.load_state_dict(extract_state_dict(agent_state_dict, 'tokenizer'))
        if load_world_model:
            self.world_model.load_state_dict(extract_state_dict(agent_state_dict, 'world_model'))
        if load_actor_critic:
            self.actor_critic.load_state_dict(extract_state_dict(agent_state_dict, 'actor_critic'))

    def act(self, obs: torch.FloatTensor, should_sample: bool = True, temperature: float = 1.0) -> torch.LongTensor:
        '''
        obs: obs转换为0～1之间的浮点数,当前的观察，n c h w
        shdoule_sample: 是否采样动作，collect中为true
        temperature: 采样温度，collect中为1.0
        '''
        # 根据配置文件是否使用原始观察还是编码重建观察来选择输入
        # 如果使用原始观察，则直接使用obs，否则对obs进行编码重建
        # 这里的obs是一个batch，shape is (N, C, H, W)
        input_ac = obs if self.actor_critic.use_original_obs else torch.clamp(self.tokenizer.encode_decode(obs, should_preprocess=True, should_postprocess=True), 0, 1)
        # actor_critic返回的是ActorCriticOutput，预测的动作和价值
        # temperature 用于控制采样的随机性，temperature越大越随机，越小则越确定
        logits_actions = self.actor_critic(input_ac).logits_actions[:, -1] / temperature
        # 根据logits_actions采样动作或选择最大概率的动作
        act_token = Categorical(logits=logits_actions).sample() if should_sample else logits_actions.argmax(dim=-1)
        return act_token # shape is (N, 1)
