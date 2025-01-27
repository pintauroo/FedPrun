# federated_experiments.py

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.utils.prune as prune
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
import numpy as np
import copy
from torchvision import models
import os
import argparse
from typing import List, Dict
import json
from datetime import datetime

# ----------------------------------------------------
# 1. CREATE VGG16 MODEL FOR CIFAR-10
# ----------------------------------------------------
def create_vgg16_for_cifar10(num_classes=10):
    """
    Creates a VGG16 model modified for CIFAR-10.
    """
    vgg = models.vgg16(pretrained=False)
    vgg.classifier[6] = nn.Linear(4096, num_classes)
    return vgg

# ----------------------------------------------------
# 2. DATASET LOADING AND PARTITIONING
# ----------------------------------------------------
def load_cifar10_data(batch_size=128, download=True, data_root="./data"):
    """
    Loads CIFAR-10 dataset with appropriate transformations.
    """
    transform_train = transforms.Compose([
        transforms.Resize(224),
        transforms.RandomCrop(224, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])

    train_set = torchvision.datasets.CIFAR10(root=data_root, train=True,
                                             download=download, transform=transform_train)
    test_set = torchvision.datasets.CIFAR10(root=data_root, train=False,
                                            download=download, transform=transform_test)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=2)

    return train_set, test_set, train_loader, test_loader

def partition_dataset_iid(dataset, num_clients=8):
    """
    Splits the dataset into IID subsets for each client.
    """
    data_len = len(dataset)
    indices = np.arange(data_len)
    np.random.shuffle(indices)

    client_indices = np.array_split(indices, num_clients)
    subsets = [Subset(dataset, idx) for idx in client_indices]
    return subsets

def partition_dataset_noniid(dataset, num_clients=8, num_classes_per_client=2):
    """
    Splits the dataset into non-IID subsets for each client, each client has data from 'num_classes_per_client' classes.

    Args:
        dataset (torchvision.datasets): The dataset to partition.
        num_clients (int): Number of clients.
        num_classes_per_client (int): Number of classes per client.

    Returns:
        List[Subset]: List of dataset subsets for each client.
    """
    # Get all unique classes
    classes = np.unique(dataset.targets)
    num_classes = len(classes)

    if num_classes_per_client > num_classes:
        raise ValueError("Number of classes per client cannot exceed total number of classes.")

    # Assign classes to clients
    client_classes = {i: np.random.choice(classes, num_classes_per_client, replace=False) for i in range(num_clients)}

    # Assign data indices to clients
    client_indices = {i: [] for i in range(num_clients)}
    for idx, target in enumerate(dataset.targets):
        for client_id, assigned_classes in client_classes.items():
            if target in assigned_classes:
                client_indices[client_id].append(idx)
                break  # Assign to the first client that has the class

    # Handle any clients with no data by assigning them random data
    for client_id in range(num_clients):
        if len(client_indices[client_id]) == 0:
            # Assign random indices to ensure every client has some data
            random_idx = np.random.choice(len(dataset), size=100, replace=False)
            client_indices[client_id].extend(random_idx.tolist())

    # Create subsets
    subsets = [Subset(dataset, client_indices[i]) for i in range(num_clients)]
    return subsets

# ----------------------------------------------------
# 3. AVERAGE WEIGHTS FUNCTION
# ----------------------------------------------------
def average_weights(client_weights: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """
    Averages the weights from multiple client models.

    Args:
        client_weights (List[Dict[str, torch.Tensor]]): List of state_dicts from clients.

    Returns:
        Dict[str, torch.Tensor]: Averaged state_dict.
    """
    avg_weights = copy.deepcopy(client_weights[0])
    for key in avg_weights.keys():
        for i in range(1, len(client_weights)):
            avg_weights[key] += client_weights[i][key]
        avg_weights[key] = torch.div(avg_weights[key], len(client_weights))
    return avg_weights

# ----------------------------------------------------
# 4. FUNCTION TO MEASURE MODEL SIZE AFTER PRUNING
# ----------------------------------------------------
def get_model_size(model: nn.Module) -> int:
    """
    Calculates the size of the model in bytes by summing the sizes of non-zero parameters and buffers.

    Args:
        model (nn.Module): The neural network model.

    Returns:
        int: Total size of the model in bytes (only non-zero parameters).
    """
    size_in_bytes = 0
    for param in model.parameters():
        # Count only non-zero elements
        non_zero_elements = torch.count_nonzero(param)
        size_in_bytes += non_zero_elements.item() * param.element_size()
    for buffer in model.buffers():
        # Count only non-zero elements in buffers
        non_zero_elements = torch.count_nonzero(buffer)
        size_in_bytes += non_zero_elements.item() * buffer.element_size()
    return size_in_bytes

# ----------------------------------------------------
# 5. FEDERATED TRAINER CLASS
# ----------------------------------------------------
class FederatedTrainer:
    def __init__(
        self,
        args,
        train_set,
        test_set,
        client_loaders: List[DataLoader],
        test_loader: DataLoader,
        device: torch.device,
        pruning_policy: str = None,
        pruning_amount: float = 0.0,
    ):
        self.args = args
        self.train_set = train_set
        self.test_set = test_set
        self.client_loaders = client_loaders
        self.test_loader = test_loader
        self.device = device
        self.pruning_policy = pruning_policy
        self.pruning_amount = pruning_amount

        # Initialize global model
        self.global_model = create_vgg16_for_cifar10(num_classes=10).to(self.device)
        if not os.path.exists(self.args.initial_model_path):
            raise FileNotFoundError(f"{self.args.initial_model_path} not found. Please run initial_training.py first to generate it.")
        self.global_model.load_state_dict(torch.load(self.args.initial_model_path, map_location='cpu'))
        self.global_model.to(self.device)
        print(f"Loaded initial model from {self.args.initial_model_path}")

        # Apply global pruning if it exists
        if self.pruning_policy in ["GUP", "LUP", "LSP"]:
            self.apply_pruning(self.global_model, self.pruning_policy, self.pruning_amount)
            print(f"Applied initial global pruning: {self.pruning_policy} with amount={self.pruning_amount}")

        # Measure model size before pruning
        self.model_size_before_pruning = get_model_size(self.global_model)
        print(f"Model size BEFORE pruning: {self.model_size_before_pruning / (1024 ** 2):.2f} MB")

        # Evaluate baseline before training
        print("\n=== Evaluating Global Model Before Federated Training ===")
        baseline_loss, baseline_acc = self.test()
        print(f"Baseline model -> Test Loss: {baseline_loss:.4f}, Test Acc: {baseline_acc:.4f}")

    def apply_pruning(self, model: nn.Module, policy: str, amount: float):
        """
        Applies the specified pruning policy to the model and removes pruning reparameterization.

        Args:
            model (nn.Module): The model to prune.
            policy (str): Pruning policy ("GUP", "LUP", "LSP").
            amount (float): Amount of pruning to apply.
        """
        if policy == "GUP":
            parameters_to_prune = []
            for name, module in model.named_modules():
                if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                    parameters_to_prune.append((module, 'weight'))

            prune.global_unstructured(
                parameters_to_prune,
                pruning_method=prune.L1Unstructured,
                amount=amount
            )
        elif policy == "LUP":
            for name, module in model.named_modules():
                if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                    prune.l1_unstructured(module, name='weight', amount=amount)
        elif policy == "LSP":
            for name, module in model.named_modules():
                if isinstance(module, nn.Conv2d):
                    prune.ln_structured(module, name='weight', amount=amount, n=2, dim=0)
                elif isinstance(module, nn.Linear):
                    prune.ln_structured(module, name='weight', amount=amount, n=2, dim=0)
        else:
            raise ValueError(f"Unknown pruning policy: {policy}")

        # Remove pruning reparameterization to avoid state_dict issues
        self.remove_pruning_reparameterization(model)

    def remove_pruning_reparameterization(self, model: nn.Module):
        """
        Removes pruning reparameterization from the model to ensure state_dict contains only standard keys.

        Args:
            model (nn.Module): The model from which to remove pruning reparameterization.
        """
        for name, module in model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                try:
                    prune.remove(module, 'weight')
                    # print(f"Removed pruning reparameterization from {name}.weight")
                except ValueError:
                    # If the module is not pruned, skip
                    pass

    def _train_client(
        self, root_model: nn.Module, train_loader: DataLoader, client_idx: int
    ) -> Dict[str, torch.Tensor]:
        """Train a client model.

        Args:
            root_model (nn.Module): server model.
            train_loader (DataLoader): client data loader.
            client_idx (int): client index.

        Returns:
            Dict[str, torch.Tensor]: client model state_dict.
        """
        model = copy.deepcopy(root_model)
        model.to(self.device)
        model.train()
        optimizer = optim.SGD(
            model.parameters(), lr=self.args.lr, momentum=self.args.momentum
        )
        criterion = nn.CrossEntropyLoss()

        epoch_loss = 0.0
        epoch_correct = 0
        epoch_samples = 0

        for epoch in range(self.args.n_client_epochs):
            for idx, (data, target) in enumerate(train_loader):
                data, target = data.to(self.device), target.to(self.device)
                optimizer.zero_grad()

                outputs = model(data)
                loss = criterion(outputs, target)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item() * data.size(0)
                _, predicted = outputs.max(1)
                epoch_correct += predicted.eq(target).sum().item()
                epoch_samples += data.size(0)

            avg_epoch_loss = epoch_loss / epoch_samples
            epoch_acc = epoch_correct / epoch_samples

            print(
                f"Client #{client_idx} | Epoch: {epoch+1}/{self.args.n_client_epochs} | Loss: {avg_epoch_loss:.4f} | Acc: {epoch_acc:.4f}",
                end="\r",
            )
        
        print()  # For newline after client training
        avg_loss = epoch_loss / (len(train_loader.dataset) * self.args.n_client_epochs)

        # Apply local pruning
        if self.pruning_policy in ["GUP", "LUP", "LSP"]:
            self.apply_pruning(model, self.pruning_policy, self.pruning_amount)
            print(f"Client #{client_idx} applied pruning: {self.pruning_policy} with amount={self.pruning_amount}")

        return model.state_dict(), avg_loss

    def train(self) -> Dict[str, any]:
        """Train the server model using federated learning.

        Returns:
            Dict[str, any]: Collected metrics from training.
        """
        train_losses = []
        self.reached_target_at = None
        metrics = {
            "pruning_policy": self.pruning_policy if self.pruning_policy else "No Pruning",
            "pruning_amount": self.pruning_amount,
            "model_size_before_pruning_MB": self.model_size_before_pruning / (1024 ** 2),
            "model_size_after_pruning_MB": get_model_size(self.global_model) / (1024 ** 2),
            "rounds": []
        }

        for epoch in range(1, self.args.n_epochs + 1):
            clients_models = []
            clients_losses = []

            print(f"\n=== Federated Round {epoch}/{self.args.n_epochs} ===")

            # Train all clients
            for client_idx in range(self.args.num_clients):
                print(f"\n--- Training on Client {client_idx+1}/{self.args.num_clients} ---")
                client_model_state, client_loss = self._train_client(
                    root_model=self.global_model,
                    train_loader=self.client_loaders[client_idx],
                    client_idx=client_idx
                )
                clients_models.append(client_model_state)
                clients_losses.append(client_loss)

            # Average client models to update global model
            updated_weights = average_weights(clients_models)
            self.global_model.load_state_dict(updated_weights)
            print("Averaged client models to update the global model.")

            # Apply global pruning
            if self.pruning_policy in ["GUP", "LUP", "LSP"]:
                self.apply_pruning(self.global_model, self.pruning_policy, self.pruning_amount)
                print(f"Applied global pruning: {self.pruning_policy} with amount={self.pruning_amount}")

                # Measure model size after pruning in each round
                current_model_size = get_model_size(self.global_model)
                print(f"Model size after global pruning in round {epoch}: {current_model_size / (1024 ** 2):.2f} MB")
            else:
                current_model_size = get_model_size(self.global_model)

            # Update average loss of this round
            avg_loss = sum(clients_losses) / len(clients_losses)
            train_losses.append(avg_loss)

            # Logging and evaluation
            total_loss, total_acc = self.test()
            avg_train_loss = sum(train_losses[-self.args.log_every:]) / min(self.args.log_every, len(train_losses))

            # Collect round metrics
            round_metrics = {
                "round": epoch,
                "avg_train_loss": avg_train_loss,
                "test_loss": total_loss,
                "test_accuracy": total_acc,
                "model_size_MB": current_model_size / (1024 ** 2)
            }
            metrics["rounds"].append(round_metrics)

            # Print results to CLI
            print(f"\n\nResults after {epoch} rounds of training:")
            print(f"---> Avg Training Loss: {avg_train_loss:.4f}")
            print(
                f"---> Avg Test Loss: {total_loss:.4f} | Avg Test Accuracy: {total_acc:.4f}\n"
            )

            # Early stopping
            if self.args.early_stopping and total_acc >= self.args.target_acc and self.reached_target_at is None:
                self.reached_target_at = epoch
                round_metrics["reached_target_at"] = self.reached_target_at
                print(
                    f"\n -----> Target accuracy {self.args.target_acc} reached at round {epoch}! <----- \n"
                )
                print(f"\nEarly stopping at round #{epoch}...")
                break

        return metrics

    def test(self) -> (float, float):
        """Test the server model.

        Returns:
            Tuple[float, float]: average loss, average accuracy.
        """
        self.global_model.eval()

        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        criterion = nn.CrossEntropyLoss()

        with torch.no_grad():
            for idx, (data, target) in enumerate(self.test_loader):
                data, target = data.to(self.device), target.to(self.device)
                outputs = self.global_model(data)
                loss = criterion(outputs, target)

                total_loss += loss.item() * data.size(0)
                _, predicted = outputs.max(1)
                total_correct += predicted.eq(target).sum().item()
                total_samples += data.size(0)

        avg_loss = total_loss / total_samples
        avg_acc = total_correct / total_samples

        return avg_loss, avg_acc

# ----------------------------------------------------
# 6. MAIN FUNCTION
# ----------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Federated Learning with Dynamic Pruning and IID/Non-IID Data Partitioning")
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training and testing')
    parser.add_argument('--n_epochs', type=int, default=10, help='Number of federated training rounds')
    parser.add_argument('--n_client_epochs', type=int, default=3, help='Number of local epochs per client')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate for the optimizer')
    parser.add_argument('--momentum', type=float, default=0.9, help='Momentum for SGD optimizer')
    parser.add_argument('--num_clients', type=int, default=4, help='Total number of federated clients')
    parser.add_argument('--frac', type=float, default=0.2, help='Fraction of clients to participate in each round')
    parser.add_argument('--pruning_amount', type=float, default=0.5, help='Amount of pruning to apply (e.g., 0.2 for 20%)')
    parser.add_argument('--global_pruning_threshold', type=float, default=0.5, help='Fraction of clients that must prune a neuron to be globally pruned')
    parser.add_argument('--log_every', type=int, default=5, help='Frequency of logging and evaluation')
    parser.add_argument('--target_acc', type=float, default=0.8, help='Target accuracy for early stopping')
    parser.add_argument('--early_stopping', action='store_true', help='Enable early stopping when target accuracy is reached')
    parser.add_argument('--data_root', type=str, default="./data", help='Root directory for dataset')
    parser.add_argument('--initial_model_path', type=str, default="initial_model.pth", help='Path to the initial model weights')
    parser.add_argument('--partB_indices_path', type=str, default="partB_indices.npy", help='Path to Part B indices file')
    parser.add_argument('--federated_model_save_dir', type=str, default="federated_models", help='Directory to save federated models')
    parser.add_argument('--experiments_results_path', type=str, default="experiments_results.json", help='Path to save experiments metrics')
    parser.add_argument('--noniid', action='store_true', help='Enable non-IID data partitioning among clients')

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create directory to save federated models
    os.makedirs(args.federated_model_save_dir, exist_ok=True)

    # Load CIFAR-10 dataset
    train_set, test_set, train_loader, test_loader = load_cifar10_data(batch_size=args.batch_size, data_root=args.data_root)

    # Load partB_indices from the saved file
    if not os.path.exists(args.partB_indices_path):
        raise FileNotFoundError(f"{args.partB_indices_path} not found. Please run initial_training.py first to generate it.")
    partB_indices = np.load(args.partB_indices_path)
    print(f"Loaded Part B indices from {args.partB_indices_path}")

    partB_subset = Subset(train_set, partB_indices)

    # Partition the dataset for federated learning (Part B)
    if args.noniid:
        print("Partitioning data in a non-IID manner among clients.")
        # You can adjust 'num_classes_per_client' as needed
        client_subsets = partition_dataset_noniid(partB_subset, num_clients=args.num_clients, num_classes_per_client=2)
    else:
        print("Partitioning data in an IID manner among clients.")
        client_subsets = partition_dataset_iid(partB_subset, num_clients=args.num_clients)
    
    client_loaders = [
        DataLoader(cs, batch_size=args.batch_size, shuffle=True, num_workers=2)
        for cs in client_subsets
    ]

    # Define pruning policies to experiment with
    pruning_policies = [None, "GUP", "LUP", "LSP"]  # None represents no pruning
    # pruning_policies = ["GUP", "LUP", "LSP"]  # None represents no pruning

    # Initialize list to collect all experiments metrics
    all_experiments_metrics = []

    # Iterate over each pruning policy and run federated training
    for policy in pruning_policies:
        if policy is None:
            pruning_policy = "No Pruning"
            pruning_amount = 0.0
        else:
            pruning_policy = policy
            pruning_amount = args.pruning_amount

        print(f"\n\n=== Starting Experiment: Pruning Policy = {pruning_policy} ===")

        # Initialize the FederatedTrainer
        trainer = FederatedTrainer(
            args=args,
            train_set=train_set,
            test_set=test_set,
            client_loaders=client_loaders,
            test_loader=test_loader,
            device=device,
            pruning_policy=policy,
            pruning_amount=pruning_amount
        )

        # Federated training with or without pruning
        print(f"\n=== Federated Learning Begins: Policy={pruning_policy}, Amount={pruning_amount}, Rounds={args.n_epochs} ===")
        metrics = trainer.train()

        # Evaluate the federated (aggregated) model
        fed_loss, fed_acc = trainer.test()
        print(f"\nFinal Federated model -> Test Loss: {fed_loss:.4f}, Test Acc: {fed_acc:.4f}")

        # Save the federated model
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        federated_model_filename = f"federated_model_{pruning_policy.replace(' ', '_')}_{timestamp}.pth"
        federated_model_path = os.path.join(args.federated_model_save_dir, federated_model_filename)
        torch.save(trainer.global_model.state_dict(), federated_model_path)
        print(f"Federated model saved to {federated_model_path}")

        # Append final metrics
        metrics["final_test_loss"] = fed_loss
        metrics["final_test_accuracy"] = fed_acc
        metrics["federated_model_path"] = federated_model_path

        # Add to all experiments
        all_experiments_metrics.append(metrics)

    # Save all experiments metrics to a JSON file
    with open(args.experiments_results_path, 'w') as f:
        json.dump(all_experiments_metrics, f, indent=4)
    print(f"\nAll experiments metrics saved to {args.experiments_results_path}")

    print("\nAll federated training experiments completed.")

if __name__ == "__main__":
    main()
