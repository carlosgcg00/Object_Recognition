# src/utils/seeds.py
import random
import torch
import numpy as np



def seed_everything(seed: int = 42, deterministic: bool = True) -> torch.Generator:
    """
    Sets global seeds outside the Dataset and returns a torch.Generator for DataLoader reproducibility.

    Args:
        seed (int): Random seed for reproducibility.
        deterministic (bool): If True, configures cuDNN for deterministic behavior.

    Returns:
        torch.Generator: Seeded generator to pass to DataLoader.
    """
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    generator = torch.Generator()
    generator.manual_seed(seed)

    return generator


def seed_worker(worker_id: int) -> None:
    """
    Initializes NumPy and Python random seeds inside each DataLoader worker.

    This is necessary when using num_workers > 0 because each worker has its own
    process and therefore its own RNG state.

    Args:
        worker_id (int): DataLoader worker ID.
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)