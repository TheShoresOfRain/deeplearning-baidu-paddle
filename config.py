__all__ = ['CONFIG', 'get']

CONFIG = {
    'model_save_dir': "./output/zodiac",
    'num_classes': 12,
    'total_images': 7096,
    'epochs': 10,
    'batch_size': 32,
    'image_shape': [3, 224, 224],
    'LEARNING_RATE': {
        'params': {
            'lr': 0.00375
        }
    },
    'OPTIMIZER': {
        'params': {
            'momentum': 0.9
        },
        'regularizer': {
            'function': 'L2',
            'factor': 0.000001
        }
    },
    'LABEL_MAP': [
        "ratt",
        "ox",
        "tiger",
        "rabbit",
        "dragon",
        "snake",
        "horse",
        "goat",
        "monkey",
        "rooster",
        "dog",
        "pig",
    ]
}


def get(full_path):
    for id, name in enumerate(full_path.split('.')):
        if id == 0:
            config = CONFIG

        config = config[name]

    return config