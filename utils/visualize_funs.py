import matplotlib.pyplot as plt
import torch
from config import cfg
import numpy as np

def visualize_fed_backdoor(fed_results, malicious_loaders):
    """
    Visualize federated learning results with backdoor attacks.

    Args:
        fed_results: Dictionary with federated results
        malicious_loaders: Dictionary of malicious client loaders
    """
    print("\n=== Federated Learning with Backdoor Attacks ===")
    
    # Extract results
    fed_global_acc = fed_results['global_clean_accuracies']
    fed_client_acc = fed_results['client_accuracies']
    fed_asr = fed_results['global_asr_values']
    
    # Check if additional metrics are available
    has_global_loss = 'global_losses' in fed_results
    has_client_loss = 'client_losses' in fed_results
    has_client_asr = 'client_asr_values' in fed_results  
    
    
    if has_global_loss:
        fed_global_loss = fed_results['global_losses']
    if has_client_loss:
        fed_client_loss = fed_results['client_losses']
    if has_client_asr:
        fed_client_asr = fed_results['client_asr_values']
    
    # Get configuration
    global_rounds = len(fed_global_acc)  # Infer from results instead of cfg
    
    # Main dashboard plots (4 subplots)
    plt.figure(figsize=(15, 15))
    
    # Plot 1: Global model accuracy
    plt.subplot(2, 2, 1)
    plt.plot(range(1, global_rounds+1), fed_global_acc, 'b-', label='Fed Global Model')
    plt.xlabel('Communication Rounds')
    plt.ylabel('Clean Test Accuracy (%)')
    plt.title('Fed: Clean Accuracy')
    plt.legend()
    plt.grid(True)
    
    # Plot 2: Attack Success Rate
    plt.subplot(2, 2, 2)
    plt.plot(range(1, global_rounds+1), fed_asr, 'r--', label='Fed ASR')
    plt.xlabel('Communication Rounds')
    plt.ylabel('Attack Success Rate (%)')
    plt.title('Fed Backdoor ASR')
    plt.legend()
    plt.grid(True)
    
    # Plot 3: Individual client model accuracies as box plots
    plt.subplot(2, 2, 3)
    
    # Prepare data for box plots
    fed_round_data = [[] for _ in range(global_rounds)]
    
    for client_idx in fed_client_acc.keys():
        client_data = fed_client_acc[client_idx]
        for round_idx in range(min(global_rounds, len(client_data))):
            fed_round_data[round_idx].append(client_data[round_idx])
    
    # Select rounds to show (first, middle, last)
    # Make sure we have data for these rounds
    available_rounds = [i for i in range(global_rounds) if fed_round_data[i]]
    if len(available_rounds) >= 3:
        rounds_to_show = [available_rounds[0], 
                          available_rounds[len(available_rounds)//2], 
                          available_rounds[-1]]
    else:
        rounds_to_show = available_rounds
    
    positions = np.array(range(1, len(rounds_to_show)+1))
    
    if rounds_to_show:  # Only create boxplot if we have rounds to show
        fed_box = plt.boxplot([fed_round_data[r] for r in rounds_to_show],
                              positions=positions,
                              widths=0.4,
                              patch_artist=True)
        for box in fed_box['boxes']:
            box.set_facecolor('lightblue')
        
        plt.xticks(positions, [f'Round {r+1}' for r in rounds_to_show])
    
    plt.ylabel('Client Test Accuracy (%)')
    plt.title('Fed: Client Accuracy Distribution')
    plt.grid(True)
    
    # Plot 4: Client distribution    
    plt.subplot(2, 2, 4)
    
    # Separate clean and malicious clients
    malicious_indices = list(malicious_loaders.keys())
    clean_indices = [i for i in fed_client_acc.keys() if i not in malicious_indices]
    
    # Get the last available round for each client type
    clean_values = []
    for client_idx in clean_indices:
        if fed_client_acc[client_idx]:  # Check if not empty
            clean_values.append(fed_client_acc[client_idx][-1])  # Get last available value
    
    malicious_values = []
    for client_idx in malicious_indices:
        if client_idx in fed_client_acc and fed_client_acc[client_idx]:  # Check if exists and not empty
            malicious_values.append(fed_client_acc[client_idx][-1])  # Get last available value
    
    # Calculate averages if data is available
    fed_clean_avg = np.mean(clean_values) if clean_values else 0
    fed_malicious_avg = np.mean(malicious_values) if malicious_values else 0
    
    # Create bar chart
    bar_width = 0.5
    bar_positions = np.array([1, 2])
    
    plt.bar(bar_positions, [fed_clean_avg, fed_malicious_avg],
            width=bar_width, label='Fed', color=['lightblue', 'lightcoral'])
    
    plt.xticks(bar_positions, ['Clean Clients', 'Malicious Clients'])
    plt.ylabel('Average Accuracy (%)')
    plt.title('Final Round: Clean vs Malicious Client Performance')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('fed_backdoor_results.png')
    
    # ADDITIONAL PLOT 1: Individual client model accuracies over rounds
    plt.figure(figsize=(12, 8))
    
    # Plot line for each client
    for client_idx, accuracies in fed_client_acc.items():
        rounds = range(1, len(accuracies) + 1)
        if client_idx in malicious_indices:
            plt.plot(rounds, accuracies, 'r-', alpha=0.7, linewidth=1.5, 
                     label=f'Malicious Client {client_idx}' if client_idx == malicious_indices[0] else "")
        else:
            plt.plot(rounds, accuracies, 'b-', alpha=0.7, linewidth=1.5,
                     label=f'Clean Client {client_idx}' if client_idx == clean_indices[0] else "")
    
    # Add global model accuracy for comparison
    plt.plot(range(1, global_rounds+1), fed_global_acc, 'g-', linewidth=2.5, label='Global Model')
    
    plt.xlabel('Communication Rounds')
    plt.ylabel('Test Accuracy (%)')
    plt.title('Individual Client Model Accuracy Tracking')
    plt.grid(True)
    
    # Create custom legend to avoid duplicate entries
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), loc='lower right')
    
    plt.tight_layout()
    plt.savefig('client_accuracy_tracking.png')
    
    # ADDITIONAL PLOT 2: Client ASR values (if available)
    if has_client_asr:
        plt.figure(figsize=(12, 8))
        
        # Plot ASR for each client
        for client_idx, asr_values in fed_client_asr.items():
            rounds = range(1, len(asr_values) + 1)
            if client_idx in malicious_indices:
                plt.plot(rounds, asr_values, 'r-', alpha=0.7, linewidth=1.5, 
                        label=f'Malicious Client {client_idx}' if client_idx == malicious_indices[0] else "")
            else:
                plt.plot(rounds, asr_values, 'b-', alpha=0.7, linewidth=1.5,
                        label=f'Clean Client {client_idx}' if client_idx == clean_indices[0] else "")
        
        # Add global ASR for comparison
        plt.plot(range(1, global_rounds+1), fed_asr, 'g-', linewidth=2.5, label='Global Model')
        
        plt.xlabel('Communication Rounds')
        plt.ylabel('Attack Success Rate (%)')
        plt.title('Individual Client Model ASR Tracking')
        plt.grid(True)
        
        # Create custom legend
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys(), loc='lower right')
        
        plt.tight_layout()
        plt.savefig('client_asr_tracking.png')
    
    # In your loss tracking plot, add this modification to check loss values and adjust visualization
    if has_global_loss or has_client_loss:
        plt.figure(figsize=(12, 8))
        
        # Print loss values for debugging
        if has_global_loss:
            print("Global loss values:", fed_global_loss)
        if has_client_loss:
            for client_idx, losses in fed_client_loss.items():
                print(f"Client {client_idx} loss values:", losses)
        
        # Plot with different line styles and increased visibility
        if has_global_loss:
            plt.plot(range(1, len(fed_global_loss)+1), fed_global_loss, 'g-', 
                    linewidth=2.5, label='Global Model Loss')
        
        if has_client_loss:
            for client_idx, losses in fed_client_loss.items():
                if len(losses) > 0:  # Ensure there's data to plot
                    rounds = range(1, len(losses) + 1)
                    line_style = 'dashed' if client_idx in malicious_indices else 'dotted'
                    if client_idx in malicious_indices:
                        plt.plot(rounds, losses, color='red', linestyle=line_style, 
                                linewidth=2.0, alpha=1.0,  # Increased visibility
                                label=f'Malicious Client {client_idx}')
                    else:
                        plt.plot(rounds, losses, color='blue', linestyle=line_style, 
                                linewidth=2.0, alpha=1.0,  # Increased visibility
                                label=f'Clean Client {client_idx}')
        
        plt.xlabel('Communication Rounds')
        plt.ylabel('Loss')
        plt.title('Loss Tracking')
        plt.grid(True)
        
        # Add points at each data point for better visibility
        if has_client_loss:
            for client_idx, losses in fed_client_loss.items():
                if len(losses) > 0:
                    rounds = range(1, len(losses) + 1)
                    color = 'red' if client_idx in malicious_indices else 'blue'
                    plt.scatter(rounds, losses, color=color, s=40)
        
        # Force y-axis range to ensure all lines are visible
        all_losses = []
        if has_global_loss:
            all_losses.extend(fed_global_loss)
        if has_client_loss:
            for losses in fed_client_loss.values():
                all_losses.extend(losses)
        
        if all_losses:
            min_loss = min(all_losses) * 0.9  # Add 10% padding
            max_loss = max(all_losses) * 1.1  # Add 10% padding
            plt.ylim(min_loss, max_loss)
        
        plt.legend(loc='upper right')
        plt.tight_layout()
        plt.savefig('loss_tracking.png')
        
        print(f"Clean accuracy: {fed_global_acc[-1]:.2f}%")
        print(f"Attack success rate: {fed_asr[-1]:.2f}%")
    
    result = {
        'plot_filename': 'fed_backdoor_results.png',
        'client_tracking_filename': 'client_accuracy_tracking.png'
    }
    
    if has_client_asr:
        result['client_asr_filename'] = 'client_asr_tracking.png'
    
    if has_global_loss or has_client_loss:
        result['loss_tracking_filename'] = 'loss_tracking.png'
    
    return result