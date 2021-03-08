import wandb
import yaml
wandb.init()
print(wandb.config)
wandb.join()