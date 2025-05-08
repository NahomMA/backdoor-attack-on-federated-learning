import torch

def evaluate_backdoor_performance(model, clean_test_loader, trigger_gen, cfg, device):
    """
    Evaluates both clean accuracy and backdoor effectiveness based on the trigger type
    specified in the configuration.

    Args:
        model: Model to evaluate
        clean_test_loader: DataLoader with clean test data
        trigger_gen: TriggerGenerator instance
        cfg: Configuration dictionary with attack parameters
        device: Computing device

    Returns:
        Dictionary with metrics: clean_acc and ASR metrics
    """
    model.eval()
    results = {
        'clean_acc': 0.0,
    }

    target_label = cfg['target_label']
  

    # 1. Evaluate Clean Accuracy
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in clean_test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)

    results['clean_acc'] = 100.0 * correct / total

    asr_sub = 0.0
    samples_tested = 0

    with torch.no_grad():
        for data, _ in clean_test_loader:
            if samples_tested >= 1000:
                break

            batch_size = data.size(0)
            data = data.to(device)

            # Apply cross trigger
            triggered_data = data.clone()
            for i in range(batch_size):
                triggered_data[i] = trigger_gen.formulate_trigger(
                    triggered_data[i]  )           
              
            # Forward pass
            output = model(triggered_data)
            pred = output.argmax(dim=1)

            # Count successful attacks
            target_labels = torch.full((batch_size,), target_label, dtype=torch.long, device=device)
            asr_sub += (pred == target_labels).sum().item()
            samples_tested += batch_size

        if samples_tested > 0:
            results['asr_global'] = 100.0 * asr_sub / samples_tested

    return results