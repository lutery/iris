from dataclasses import dataclass
from typing import Any, Optional, Tuple

from einops import rearrange
import torch
import torch.nn as nn
import torch.nn.functional as F

from dataset import Batch
from .kv_caching import KeysValues
from .slicer import Embedder, Head
from .tokenizer import Tokenizer
from .transformer import Transformer, TransformerConfig
from utils import init_weights, LossWithIntermediateLosses


@dataclass
class WorldModelOutput:
    output_sequence: torch.FloatTensor # ransformer 模型的完整输出序列，包含所有位置的特征表示，包含一个时间步的tokens和一个动作
    logits_observations: torch.FloatTensor # 预测下一个观察 token 的未归一化概率分布（logits）
    logits_rewards: torch.FloatTensor # 预测下一个奖励 token 的未归一化概率分布（logits）
    logits_ends: torch.FloatTensor # 预测下一个结束 token 的未归一化概率分布（logits）


class WorldModel(nn.Module):
    def __init__(self, obs_vocab_size: int, act_vocab_size: int, config: TransformerConfig) -> None:
        '''
        obs_vocab_size: int, 观察的环境提取的特征维度
        act_vocab_size: int, 动作的特征维度
        config: 世界模型的配置参数，TransformerConfig类型
            _target_: models.TransformerConfig
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
        self.obs_vocab_size, self.act_vocab_size = obs_vocab_size, act_vocab_size
        self.config = config
        self.transformer = Transformer(config)

        # 以下的作用，为后续embedder去提取观察和动作拼接的tensor做准确
        all_but_last_obs_tokens_pattern = torch.ones(config.tokens_per_block) # 一个shape为17的全1向量，倒数第二个位置为0
        '''
        世界模型中，序列被组织为块，每个块包含：
        16个观察 tokens (obs_tokens)
        1个动作 token (act_token)
        '''
        all_but_last_obs_tokens_pattern[-2] = 0 # 掩盖掉倒数第二个位置，也就是最后一个观察 token 的位置
        '''
        [obs1, obs2, ..., obs15, obs16, act]
                      ^       ^
                      |       |
                   索引[-2]  索引[-1]
        '''
        act_tokens_pattern = torch.zeros(self.config.tokens_per_block) # 一个shape为17的全零向量，这个大小应该就是观察tokens中token的维度+动作的维度1
        act_tokens_pattern[-1] = 1 # 最后一个位置为1
        obs_tokens_pattern = 1 - act_tokens_pattern # 只有最后一个位置为0，其余位置为1

        # 位置编码器
        self.pos_emb = nn.Embedding(config.max_tokens, config.embed_dim)

        self.embedder = Embedder(
            max_blocks=config.max_blocks,
            block_masks=[act_tokens_pattern, obs_tokens_pattern],
            embedding_tables=nn.ModuleList([nn.Embedding(act_vocab_size, config.embed_dim), nn.Embedding(obs_vocab_size, config.embed_dim)])
        )

        # todo 以下三个的作用？
        # 这里应该是观察头
        self.head_observations = Head(
            max_blocks=config.max_blocks,
            block_mask=all_but_last_obs_tokens_pattern,
            head_module=nn.Sequential(
                nn.Linear(config.embed_dim, config.embed_dim),
                nn.ReLU(),
                nn.Linear(config.embed_dim, obs_vocab_size)
            )
        )

        # # 这里应该是奖励头
        self.head_rewards = Head(
            max_blocks=config.max_blocks,
            block_mask=act_tokens_pattern,
            head_module=nn.Sequential(
                nn.Linear(config.embed_dim, config.embed_dim),
                nn.ReLU(),
                nn.Linear(config.embed_dim, 3)
            )
        )

        # 于是是否中止
        self.head_ends = Head(
            max_blocks=config.max_blocks,
            block_mask=act_tokens_pattern,
            head_module=nn.Sequential(
                nn.Linear(config.embed_dim, config.embed_dim),
                nn.ReLU(),
                nn.Linear(config.embed_dim, 2)
            )
        )

        # 初始化权重
        self.apply(init_weights)

    def __repr__(self) -> str:
        return "world_model"

    def forward(self, tokens: torch.LongTensor, past_keys_values: Optional[KeysValues] = None) -> WorldModelOutput:
        '''
        tokens: 包含了观察和动作的tokens，(N, T(H/4*W/4+1))
        past_keys_values: todo 在训练world_model时传入的是None
        '''

        num_steps = tokens.size(1)  # T(H/4*W/4+1)
        assert num_steps <= self.config.max_tokens
        prev_steps = 0 if past_keys_values is None else past_keys_values.size # 在训练world_model时传入的是None，所以prev_steps为0

        # self.embedder(tokens, num_steps, prev_steps): 将tokens中观察和动作分别进行嵌入，得到一个形状为 (N, T(H/4*W/4+1), embed_dim) 的张量
        # self.pos_emb(prev_steps + torch.arange(num_steps, device=tokens.device))：根据时间步的长度，创建一个位置编码的张量，形状为 (T(H/4*W/4+1), embed_dim)
        # 将位置信息和嵌入的tokens进行相加，得到一个形状为 (N, T(H/4*W/4+1), embed_dim) 的张量
        sequences = self.embedder(tokens, num_steps, prev_steps) + self.pos_emb(prev_steps + torch.arange(num_steps, device=tokens.device))

        x = self.transformer(sequences, past_keys_values) # x shape (N, T(H/4*W/4+1), embed_dim)

        logits_observations = self.head_observations(x, num_steps=num_steps, prev_steps=prev_steps)
        logits_rewards = self.head_rewards(x, num_steps=num_steps, prev_steps=prev_steps)
        logits_ends = self.head_ends(x, num_steps=num_steps, prev_steps=prev_steps)

        return WorldModelOutput(x, logits_observations, logits_rewards, logits_ends)

    def compute_loss(self, batch: Batch, tokenizer: Tokenizer, **kwargs: Any) -> LossWithIntermediateLosses:
        '''
        batch:
        observations shape (batch_num_samples, sequence_length, channels, height, width)
        actions shape (batch_num_samples, sequence_length, action_dim)
        rewards shape (batch_num_samples, sequence_length)
        dones shape (batch_num_samples, sequence_length)
        '''
        with torch.no_grad():
            obs_tokens = tokenizer.encode(batch['observations'], should_preprocess=True).tokens  # (N, T, H/4*W/4)

        act_tokens = rearrange(batch['actions'], 'b l -> b l 1') # (N, T, 1)
        # torch.cat((obs_tokens, act_tokens), dim=2) shape (N, T, H/4*W/4+1)
        # rearrange shape (N, T, H/4*W/4+1) -> (N, T(H/4*W/4+1))
        tokens = rearrange(torch.cat((obs_tokens, act_tokens), dim=2), 'b l k1 -> b (l k1)')  # 包含了观察和动作的tokens，(N, T(H/4*W/4+1))
        outputs = self(tokens)

        # 这样看来，世界模型就是根据前N个观察特征+第N个动作特征，预测第N+1个观察特征
        # 然后根据第N个动作特征预测第N+1个奖励特征和是否终止的特征 
        # act_tokens_pattern 只决定了这些头输出的位置（哪些位置需要预测）
        # 实际的输入来自transformer的输出 x，其中已经编码了整个序列的信息
        # 由于transformer使用了自注意力机制，所以动作位置的特征表示已经融合了历史观察和动作信息
        labels_observations, labels_rewards, labels_ends = self.compute_labels_world_model(obs_tokens, batch['rewards'], batch['ends'], batch['mask_padding'])

        logits_observations = rearrange(outputs.logits_observations[:, :-1], 'b t o -> (b t) o')
        loss_obs = F.cross_entropy(logits_observations, labels_observations)
        loss_rewards = F.cross_entropy(rearrange(outputs.logits_rewards, 'b t e -> (b t) e'), labels_rewards)
        loss_ends = F.cross_entropy(rearrange(outputs.logits_ends, 'b t e -> (b t) e'), labels_ends)

        # 得到预测下一个观察、下一个奖励、下一个停止标识的损失
        return LossWithIntermediateLosses(loss_obs=loss_obs, loss_rewards=loss_rewards, loss_ends=loss_ends)

    def compute_labels_world_model(self, obs_tokens: torch.Tensor, rewards: torch.Tensor, ends: torch.Tensor, mask_padding: torch.BoolTensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        assert torch.all(ends.sum(dim=1) <= 1)  # at most 1 done
        mask_fill = torch.logical_not(mask_padding) # 将有效数据掩码 (mask_padding, 值为 True 表示有效) 转换为填充掩码 (mask_fill, 值为 True 表示需要填充)，用于后续标记需要忽略的位置。
        # mask_fill.unsqueeze(-1) - 在最后一维添加一个维度，形状变为 (N, T, 1)
        # .expand_as(obs_tokens) - 扩展掩码以匹配观察 tokens 的形状 (N, T, H/4*W/4)
        # obs_tokens.masked_fill(..., -100) - 将无效位置（掩码为 True）的值替换为 -100，这是 PyTorch 中交叉熵损失忽略的特殊值
        # rearrange(..., 'b t k -> b (t k)') - 重排形状，将每个时间步的 token 展平为一维序列，形状变为 (N, T*H/4*W/4)
        # [:, 1:] - 去除序列的第一个元素，从而实现预测下一个 token（而不是当前 token）
        labels_observations = rearrange(obs_tokens.masked_fill(mask_fill.unsqueeze(-1).expand_as(obs_tokens), -100), 'b t k -> b (t k)')[:, 1:]
        # rewards.sign() - 获取奖励的符号 (-1, 0, 1)
        # + 1 - 将范围调整为 (0, 1, 2)，适合作为分类目标
        # .masked_fill(mask_fill, -100) - 将无效位置标记为 -100 (忽略) todo 查看masked_fill是如何填充的
        # .long() - 转换为长整型，适合分类损失函数
        labels_rewards = (rewards.sign() + 1).masked_fill(mask_fill, -100).long()  # Rewards clipped to {-1, 0, 1}
        # 将无效位置的终止信号标记为 -100 (忽略)
        # 保持有效位置的值不变（0 表示继续，1 表示终止）
        labels_ends = ends.masked_fill(mask_fill, -100)
        # 将所有标签展平为一维张量，这是因为后续的损失计算使用了形如 F.cross_entropy(rearrange(outputs.logits_X, 'b t e -> (b t) e'), labels_X) 的操作。
        return labels_observations.reshape(-1), labels_rewards.reshape(-1), labels_ends.reshape(-1)
