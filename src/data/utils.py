import torch
DATA_SEED = 42

def train_test_split(data, test_fraction):
    """
    Split data into train and test sets.
    """
    n = len(data)
    indices = torch.randperm(n, generator=torch.Generator().manual_seed(DATA_SEED))
    test_size = int(n * test_fraction)
    return data[indices[:test_size]], data[indices[test_size:]]


def get_k_folds(data, k):
    """
    Split data into k folds.
    """
    n = len(data)
    valid_size = int(n / k)
    indices = torch.randperm(n, generator=torch.Generator().manual_seed(DATA_SEED))
    folds = []
    for i in range(k):
        folds.append(
            (
                indices[i * valid_size : (i + 1) * valid_size],
                torch.cat([indices[: i * valid_size], indices[(i + 1) * valid_size :]]),
            )
        )
    return folds
