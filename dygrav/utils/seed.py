import random, os
def seed_everything(seed: int = 17):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
