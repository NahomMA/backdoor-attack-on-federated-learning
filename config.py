import multiprocessing

#config.py
cfg = {
        "data_path": "dataset",
        "batch_size": 64,
        "num_workers": multiprocessing.cpu_count(),
        "shuffle": True,
        "local_epochs": 20,
        "learning_rate": 0.001,
        'global_rounds': 10,
        'num_clients': 10,
        'num_classes'
        : 10,
        'malicious_clients': 3,
        'malicious_ratio': 0.3,
        'malicious_type': 'label_flipping',
        'malicious_target': 0,
        'malicious_trigger_type': 'cross',
        'malicious_trigger_size': 4,
        'malicious_trigger_intensity': 0.5,

}

