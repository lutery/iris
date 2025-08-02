import random
from typing import List, Optional, Union

import gymnasium as gym
from einops import rearrange
import numpy as np
from PIL import Image
import torch
from torch.distributions.categorical import Categorical
import torchvision


class WorldModelEnv:

    def __init__(self, tokenizer: torch.nn.Module, world_model: torch.nn.Module, device: Union[str, torch.device], env: Optional[gym.Env] = None) -> None:

        self.device = torch.device(device)
        self.world_model = world_model.to(self.device).eval()
        self.tokenizer = tokenizer.to(self.device).eval()

        self.keys_values_wm, self.obs_tokens, self._num_observations_tokens = None, None, None

        self.env = env

    @property
    def num_observations_tokens(self) -> int:
        return self._num_observations_tokens # todo 观察在特征提取后的尺寸大小（tokens）H/4*W/4 / 但是根据实际排查有可能是序列的观察长度

    @torch.no_grad()
    def reset(self) -> torch.FloatTensor:
        assert self.env is not None
        obs = torchvision.transforms.functional.to_tensor(self.env.reset()).to(self.device).unsqueeze(0)  # (1, C, H, W) in [0., 1.]
        return self.reset_from_initial_observations(obs)

    @torch.no_grad()
    def reset_from_initial_observations(self, observations: torch.FloatTensor) -> torch.FloatTensor:
        '''
        observations shape is (B, C, H, W), 代表最后一个时间步的观察数据
        '''
        obs_tokens = self.tokenizer.encode(observations, should_preprocess=True).tokens    # (B, C, H, W) -> (B, K) = (B, H/4*W/4)
        _, num_observations_tokens = obs_tokens.shape # num_observations_tokens = K = (B, H/4*W/4)
        if self.num_observations_tokens is None:
            self._num_observations_tokens = num_observations_tokens

        '''
        当触发重置时，该调用会：

        丢弃历史上下文：放弃之前所有已经处理过的 tokens 的键值缓存
        保留当前观察：使用当前的观察 tokens (self.obs_tokens) 重新初始化上下文
        重建键值缓存：创建新的键值缓存，只包含当前观察的表示
        重新进行初始推理：用当前观察 tokens 进行一次世界模型推理
        '''
        _ = self.refresh_keys_values_with_initial_obs_tokens(obs_tokens) # todo 测试不调用的后果？，会保留之前的信息？
        self.obs_tokens = obs_tokens

        return self.decode_obs_tokens()

    @torch.no_grad()
    def refresh_keys_values_with_initial_obs_tokens(self, obs_tokens: torch.LongTensor) -> torch.FloatTensor:
        '''
        obs_tokens: shape is (B, K)  = (B, H/4*W/4)，最后一个时间步的观察数据的特征提取后的tokens
        '''
        n, num_observations_tokens = obs_tokens.shape
        assert num_observations_tokens == self.num_observations_tokens
        # 这里主要是生成一个空的键值缓存（KV Cache），用于存储Transformer的注意力机制中的键和值
        self.keys_values_wm = self.world_model.transformer.generate_empty_keys_values(n=n, max_tokens=self.world_model.config.max_tokens)
        outputs_wm = self.world_model(obs_tokens, past_keys_values=self.keys_values_wm)
        return outputs_wm.output_sequence  # (B, K, E)

    @torch.no_grad()
    def step(self, action: Union[int, np.ndarray, torch.LongTensor], should_predict_next_obs: bool = True) -> None:
        '''
        action: 可以是整数、numpy数组或torch.LongTensor，表示要执行的动作
        should_predict_next_obs: 是否预测下一个观察特征，默认为True
        '''
        assert self.keys_values_wm is not None and self.num_observations_tokens is not None

        num_passes = 1 + self.num_observations_tokens if should_predict_next_obs else 1

        output_sequence, obs_tokens = [], []

        if self.keys_values_wm.size + num_passes > self.world_model.config.max_tokens:
            # 如果超过了最大tokens数量，则刷新键值缓存，应该是去除了过于旧的tokens
            _ = self.refresh_keys_values_with_initial_obs_tokens(self.obs_tokens)

        token = action.clone().detach() if isinstance(action, torch.Tensor) else torch.tensor(action, dtype=torch.long)
        token = token.reshape(-1, 1).to(self.device)  # (B, 1) 展平动作token

        for k in range(num_passes):  # assumption that there is only one action token.
            
            # 这里的世界模型应该是已经传入了一个起始的观察特征（obs_tokens），然后开始i模拟执行动作，预测下一个观察特征
            outputs_wm = self.world_model(token, past_keys_values=self.keys_values_wm)
            output_sequence.append(outputs_wm.output_sequence)

            if k == 0: # 起始的时候
                reward = Categorical(logits=outputs_wm.logits_rewards).sample().float().cpu().numpy().reshape(-1) - 1   # (B,)
                done = Categorical(logits=outputs_wm.logits_ends).sample().cpu().numpy().astype(bool).reshape(-1)       # (B,)

            if k < self.num_observations_tokens:
                token = Categorical(logits=outputs_wm.logits_observations).sample()
                obs_tokens.append(token)

        output_sequence = torch.cat(output_sequence, dim=1)   # (B, 1 + K, E)
        self.obs_tokens = torch.cat(obs_tokens, dim=1)        # (B, K)

        obs = self.decode_obs_tokens() if should_predict_next_obs else None
        return obs, reward, done, None

    @torch.no_grad()
    def render_batch(self) -> List[Image.Image]:
        frames = self.decode_obs_tokens().detach().cpu()
        frames = rearrange(frames, 'b c h w -> b h w c').mul(255).numpy().astype(np.uint8)
        return [Image.fromarray(frame) for frame in frames]

    @torch.no_grad()
    def decode_obs_tokens(self) -> List[Image.Image]:
        '''
        将提取的连续特征，转换为离散特征，最后还原为图像（值范围0～1）
        '''
        embedded_tokens = self.tokenizer.embedding(self.obs_tokens)     # (B, K, E)
        z = rearrange(embedded_tokens, 'b (h w) e -> b e h w', h=int(np.sqrt(self.num_observations_tokens)))
        rec = self.tokenizer.decode(z, should_postprocess=True)         # (B, C, H, W)
        return torch.clamp(rec, 0, 1)

    @torch.no_grad()
    def render(self):
        assert self.obs_tokens.shape == (1, self.num_observations_tokens)
        return self.render_batch()[0]
