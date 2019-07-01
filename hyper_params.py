import pandas as pd
def get_hyper_params():
    # hyper parameters
    hyper_params = {
        "alpha": 0.2,
        "learning_rate": 1e-3,
        "batch_size": 64,
        'epoch': 70,
        'dropout': 0.2,
        # etc
        'input_shape': [32, 32, 3],
        'num_classes': 10,
        'dataset': 'cifar10'}
    return hyper_params

