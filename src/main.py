import hydra
from omegaconf import DictConfig

from trainer import Trainer


# 配置管理工具，能够自动解析yaml配置文件并将配置文件的配置转换后传给主函数
@hydra.main(config_path="../config", config_name="trainer")
def main(cfg: DictConfig):
    trainer = Trainer(cfg)
    trainer.run()


if __name__ == "__main__":
    main()
