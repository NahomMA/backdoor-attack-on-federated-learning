import multiprocessing

#config.py
cfg = {
        "data_path": "dataset",
        "batch_size": 64,
        "num_workers": multiprocessing.cpu_count(),
        "shuffle": True,
        "local_epochs": 5,
        "learning_rate": 0.0005,
        'global_rounds': 20,
        'num_clients': 10,
        'num_classes': 10,
        'malicious_clients': 3,
        'malicious_indices':[0,1,3],
        'target_label': 1,
        'poison_ratio': 0.1,
        'intensity': 0.5,
        'trigger_size': 4
        
}