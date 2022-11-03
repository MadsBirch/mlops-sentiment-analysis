import hydra






from omegaconf import DictConfig, OmegaConf
import hydra

@hydra.main(version_base=None)
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

if __name__ == "__main__":
    main()


@hydra.main(version_base=None, config_path="../../outputs/2022-11-03/13-48-50/.hydra/", config_name="config")
def train_model(cfg):
    print(OmegaConf.to_yaml(cfg))

if __name__ == "__main__":
    train_model()