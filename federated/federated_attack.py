import copy
import torch
import torch.optim as optim
from collections import defaultdict
from src.train import train_client, evaluate_model
from src.evaluate import evaluate_backdoor_performance
from src.model import LeNet

def federated_attack(trigger_gen,client_loaders, malicious_loaders,test_loader,cfg, device):
 
    # Initialize global model
    global_model = LeNet().to(device)
    malicious_client_indices = cfg['malicious_indices']

    # Track metrics
    global_clean_accuracies = []
    global_asr_values = []
    global_losses = []
    client_accuracies = defaultdict(list)
    client_losses = defaultdict(list)
    client_asr_values = defaultdict(list)

    # Create poisoned loaders for malicious clients
    poisoned_loaders = trigger_gen.create_poisoned_loaders(
        malicious_loaders=malicious_loaders )

    for round_num in range(cfg['global_rounds']):
        print(f"\nRound {round_num+1}/{cfg['global_rounds']}")

        # Store updated client models
        client_models = []
        client_sizes = []
        client_losses = defaultdict(list)
        # Client training
        description_st = ''
        for client_idx, client_loader in client_loaders.items():
            # Copy global model to client
            client_model = copy.deepcopy(global_model)
            client_optimizer = optim.SGD(client_model.parameters(), lr=cfg['learning_rate'], momentum=0.9)

            # Choose appropriate training loader
            if client_idx in malicious_client_indices:
                if client_idx in poisoned_loaders:
                    description_str = f'Training malicious client {client_idx} with poisoned data'
                    train_loader = poisoned_loaders[client_idx]
                else:
                    description_str = f'Training malicious client {client_idx} with clean data'
                    train_loader = client_loader
            else:
                description_str = f'Training clean client {client_idx} with clean data'
                train_loader = client_loader

            # Train client model
            client_model = train_client(client_model, client_optimizer, train_loader, cfg['local_epochs'], device)

            # Evaluate client model
            client_loss, client_acc = evaluate_model(client_model, test_loader,device)
            client_accuracies[client_idx].append(client_acc)
            client_losses[client_idx].append(client_loss)  
            client_backdoor_results = evaluate_backdoor_performance(
                model=client_model,
                clean_test_loader=test_loader,
                trigger_gen=trigger_gen,
                cfg=cfg,
                device=device
            )                             
            client_asr_values[client_idx].append(client_backdoor_results['asr_global'])
            print( description_str + f" accuracy: {client_acc:.2f}% , ASR: {client_backdoor_results['asr_global']:.2f}%")

            # Store client model and dataset size
            client_models.append(client_model)
            client_sizes.append(len(train_loader.dataset))

        # Perform FedAvg aggregation
        with torch.no_grad():
            # Calculate weights for each client
            total_size = sum(client_sizes)
            client_weights = [size / total_size for size in client_sizes]

            # Update global model parameters
            for param_idx, param in enumerate(global_model.parameters()):
                weighted_sum = torch.zeros_like(param)
                for client_idx, client_model in enumerate(client_models):
                    client_param = list(client_model.parameters())[param_idx]
                    weighted_sum += client_weights[client_idx] * client_param
                param.data = weighted_sum.data

        # Standard evaluation for clean accuracy
        global_loss, global_acc = evaluate_model(global_model, test_loader,device)
        global_clean_accuracies.append(global_acc)
        global_losses.append(global_loss)   
        print(f"Global model clean accuracy: {global_acc:.2f}%")

        # Evaluate backdoor performance
        print("Evaluating backdoor performance...")
        backdoor_results = evaluate_backdoor_performance(
            model=global_model,
            clean_test_loader=test_loader,
            trigger_gen=trigger_gen,
            cfg=cfg,
            device=device
        )

        # Store and display results
        global_asr_values.append(backdoor_results['asr_global'])
        print(f"Global trigger ASR: {backdoor_results['asr_global']:.2f}%")

        if 'asr_sub' in backdoor_results:
            print(f"Subtractive trigger ASR: {backdoor_results['asr_sub']:.2f}%")

        # Optionally display pattern-specific ASRs
        if 'asr_patterns' in backdoor_results and backdoor_results['asr_patterns']:
            print("Pattern-specific ASRs:")
            for pattern, asr in backdoor_results['asr_patterns'].items():
                print(f"  {pattern}: {asr:.2f}%")

    # Return both clean accuracy and backdoor metrics
    return {
        'global_clean_accuracies': global_clean_accuracies,
        'global_asr_values': global_asr_values,
        'client_accuracies': client_accuracies,
        'client_losses': client_losses,        
        'client_asr_values': client_asr_values,
        'global_losses': global_losses,
        'final_backdoor_results': backdoor_results,
    }
