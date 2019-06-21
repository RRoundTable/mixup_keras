import pandas as pd
def get_hyper_params():
    # hyper parameters
    hyper_params = {
        "alpha": 0.2,
        "learning_rate": 1e-3,
        "batch_size": 32,
        'epoch': 50,
        'dropout': 0.2,
        # etc
        'input_shape': [28, 28, 1],
        'num_classes': 10,
        'dataset': 'mnist'}
    return hyper_params

