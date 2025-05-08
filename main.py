from device import set_device_seed
from config import cfg
from data.data_loader import DataSetLoader
from data.iid_data_partition import IIDDataPartition
from attack.trigger_generator import TriggerGenerator
from federated.federated_attack import federated_attack
from utils.visualize_funs import visualize_fed_backdoor


def main():
    #set seed for reproducibility
    device = set_device_seed(42)
    train_dataset, test_dataset, test_loader = DataSetLoader(cfg).load_data()
    
    # Create IID data partition
    iid_partition = IIDDataPartition(train_dataset, cfg)
    client_loaders, malicious_loaders = iid_partition.create_client_loaders()
    iid_partition.plot_label_distribution()
    iid_partition.plot_label_client_distribution()    

    # Create trigger generator
    trigger_generator = TriggerGenerator(cfg)
    trigger_generator.visualize_triggers(client_loaders[0])

    #perform federated attack
    fed_results = federated_attack(trigger_generator,client_loaders, malicious_loaders,test_loader,cfg, device)

    #visualize the results
    visualize_fed_backdoor(fed_results, malicious_loaders)

if __name__ == "__main__":
    main()














