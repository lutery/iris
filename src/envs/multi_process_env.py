from dataclasses import astuple, dataclass
from enum import Enum
from multiprocessing import Pipe, Process
from multiprocessing.connection import Connection
from typing import Any, Callable, Iterator, List, Optional, Tuple

import numpy as np

from .done_tracker import DoneTrackerEnv


class MessageType(Enum):
    RESET = 0
    RESET_RETURN = 1
    STEP = 2
    STEP_RETURN = 3
    CLOSE = 4


@dataclass
class Message:
    type: MessageType
    content: Optional[Any] = None

    def __iter__(self) -> Iterator:
        return iter(astuple(self))


def child_env(child_id: int, env_fn: Callable, child_conn: Connection) -> None:
    '''
    child_id: 0, 1, ..., num_envs - 1 子进程id
    env_fn: Callable, 返回一个环境实例的函数
    child_conn: Connection, 子进程与父进程之间通信的管道
    该函数在子进程中运行，负责接收父进程的指令并执行相应的环境操作后，将观察反馈发送回父进程
    '''
    np.random.seed(child_id + np.random.randint(0, 2 ** 31 - 1))
    env = env_fn()
    while True:
        message_type, content = child_conn.recv() # 接收消息
        if message_type == MessageType.RESET: # 重置环境
            obs = env.reset()
            child_conn.send(Message(MessageType.RESET_RETURN, obs)) #
        elif message_type == MessageType.STEP:
            result = env.step(content)
            if len(result) == 4:
                obs, rew, done, _ = result
                truncated = False
            else:  # len(result) == 5
                obs, rew, done, truncated, _ = result
            if done or truncated:
                obs = env.reset()
            child_conn.send(Message(MessageType.STEP_RETURN, (obs, rew, done or truncated, None)))
        elif message_type == MessageType.CLOSE:
            child_conn.close()
            return
        else:
            raise NotImplementedError


class MultiProcessEnv(DoneTrackerEnv):
    def __init__(self, env_fn: Callable, num_envs: int, should_wait_num_envs_ratio: float) -> None:
        '''
        should_wait_num_envs_ratio: 1.0 todo 作用
        '''
        super().__init__(num_envs)
        self.num_actions = env_fn().env.action_space.n
        self.should_wait_num_envs_ratio = should_wait_num_envs_ratio
        self.processes, self.parent_conns = [], [] # 子进程、父进程之间通信的管道
        for child_id in range(num_envs):
            parent_conn, child_conn = Pipe() # 创建一个多进程之间通信的管道，用于父子进程之间的通信
            self.parent_conns.append(parent_conn)
            p = Process(target=child_env, args=(child_id, env_fn, child_conn), daemon=True)
            self.processes.append(p)
        for p in self.processes:
            p.start()

    def should_reset(self) -> bool:
        # 如果当前已经结束的环境数量大于等于应该等待的环境数量比例，则返回True
        # 否则返回False，在当前的默认配置中，should_wait_num_envs_ratio=1.0
        # 即所有环境都结束了才会重置
        return (self.num_envs_done / self.num_envs) >= self.should_wait_num_envs_ratio

    def _receive(self, check_type: Optional[MessageType] = None) -> List[Any]:
        messages = [parent_conn.recv() for parent_conn in self.parent_conns]
        if check_type is not None:
            assert all([m.type == check_type for m in messages])
        return [m.content for m in messages]

    def reset(self) -> np.ndarray:
        self.reset_done_tracker()
        for parent_conn in self.parent_conns:
            parent_conn.send(Message(MessageType.RESET))
        content = self._receive(check_type=MessageType.RESET_RETURN)
        return np.stack(content)

    def step(self, actions: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Any]:
        # 给每一个子进程发送动作
        for parent_conn, action in zip(self.parent_conns, actions):
            parent_conn.send(Message(MessageType.STEP, action))
        # 接收每个子进程的返回结果
        content = self._receive(check_type=MessageType.STEP_RETURN)
        obs, rew, done, _ = zip(*content)
        # 用结束标识更新done_tracker
        done = np.stack(done)
        self.update_done_tracker(done)
        return np.stack(obs), np.stack(rew), done, None

    def close(self) -> None:
        for parent_conn in self.parent_conns:
            parent_conn.send(Message(MessageType.CLOSE))
        for p in self.processes:
            p.join()
        for parent_conn in self.parent_conns:
            parent_conn.close()
