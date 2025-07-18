import numpy as np


class DoneTrackerEnv:
    '''
    跟踪每个环境的完成状态的环境基类。
    '''

    def __init__(self, num_envs: int) -> None:
        """Monitor env dones: 0 when not done, 1 when done, 2 when already done."""
        self.num_envs = num_envs
        self.done_tracker = None
        self.reset_done_tracker()

    def reset_done_tracker(self) -> None:
        # 根据环境的数量创建一个 done_tracker 数组，初始值为 0，即为False，未结束
        self.done_tracker = np.zeros(self.num_envs, dtype=np.uint8)

    def update_done_tracker(self, done: np.ndarray) -> None:
        # 更新每个环境的结束状态，这里之所以用2就是可以区分新结束的和已经结束的环境
        # 假设新结束，那么done=1， self.done_tracker = 0，最终结果1
        # 假设已经结束，那么done=1， self.done_tracker = 1，最终结果2
        self.done_tracker = np.clip(2 * self.done_tracker + done, 0, 2)

    @property
    def num_envs_done(self) -> int:
        # 返回当前已经结束的环境数量
        return (self.done_tracker > 0).sum()

    @property
    def mask_dones(self) -> np.ndarray:
        # 返回一个布尔数组，表示哪些环境已经结束，False表示结束， True表示未结束
        return np.logical_not(self.done_tracker)

    @property
    def mask_new_dones(self) -> np.ndarray:
        # self.done_tracker[self.done_tracker <= 1]：表示哪些环境是新结束的或者未结束
        return np.logical_not(self.done_tracker[self.done_tracker <= 1])
