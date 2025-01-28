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
        dataset (torch.utils.data.Dataset or Subset): The dataset or Subset to partition.
        num_clients (int): Number of clients.
        num_classes_per_client (int): Number of classes per client.

    Returns:
        List[Subset]: List of dataset subsets for each client.
    """
    # Extract targets based on whether dataset is a Subset or full dataset
    if isinstance(dataset, Subset):
        # For Subset, access the underlying dataset's targets using the subset's indices
        targets = np.array(dataset.dataset.targets)[dataset.indices]
        dataset_indices = dataset.indices
    else:
        targets = np.array(dataset.targets)
        dataset_indices = np.arange(len(dataset))

    classes = np.unique(targets)
    num_classes = len(classes)

    if num_classes_per_client > num_classes:
        raise ValueError("Number of classes per client cannot exceed total number of classes.")

    # Assign classes to clients
    client_classes = {}
    for i in range(num_clients):
        # Ensure unique class assignments across clients if possible
        available_classes = list(set(classes) - set(client_classes.get(i, [])))
        if len(available_classes) < num_classes_per_client:
            available_classes = classes
        assigned = np.random.choice(classes, num_classes_per_client, replace=False)
        client_classes[i] = assigned

    # Assign data indices to clients based on assigned classes
    client_indices = {i: [] for i in range(num_clients)}
    for idx, target in zip(dataset_indices, targets):
        # Assign to the first client that has the class
        for client_id, assigned_classes in client_classes.items():
            if target in assigned_classes:
                client_indices[client_id].append(idx)
                break

    # Handle any clients with no data by assigning them random data
    for client_id in range(num_clients):
        if len(client_indices[client_id]) == 0:
            # Assign random indices to ensure every client has some data
            if isinstance(dataset, Subset):
                random_idx = np.random.choice(dataset_indices, size=100, replace=False)
            else:
                random_idx = np.random.choice(len(dataset), size=100, replace=False)
            client_indices[client_id].extend(random_idx.tolist())

    # Create subsets
    if isinstance(dataset, Subset):
        subsets = [Subset(dataset.dataset, client_indices[i]) for i in range(num_clients)]
    else:
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
        consensus: bool = False,
        adjacency_matrix: np.ndarray = None,
    ):
        self.args = args
        self.train_set = train_set
        self.test_set = test_set
        self.client_loaders = client_loaders
        self.test_loader = test_loader
        self.device = device
        self.pruning_policy = pruning_policy
        self.pruning_amount = pruning_amount
        self.consensus = consensus
        self.adjacency_matrix = adjacency_matrix

        if self.consensus:
            # Initialize individual models for each client
            self.client_models = [
                create_vgg16_for_cifar10(num_classes=10).to(self.device)
                for _ in range(self.args.num_clients)
            ]
            if not os.path.exists(self.args.initial_model_path):
                raise FileNotFoundError(f"{self.args.initial_model_path} not found. Please run initial_training.py first to generate it.")
            initial_state = torch.load(self.args.initial_model_path, map_location='cpu')
            for model in self.client_models:
                model.load_state_dict(initial_state)
                model.to(self.device)
            print(f"Loaded initial model from {self.args.initial_model_path} to all clients.")

            # Apply global pruning if it exists
            if self.pruning_policy in ["GUP", "LUP", "LSP"]:
                for idx, model in enumerate(self.client_models):
                    self.apply_pruning(model, self.pruning_policy, self.pruning_amount)
                    print(f"Applied initial global pruning to Client {idx} using {self.pruning_policy} with amount={self.pruning_amount}")

            # Measure model size before pruning
            self.model_size_before_pruning = get_model_size(self.client_models[0])
            print(f"Model size BEFORE pruning: {self.model_size_before_pruning / (1024 ** 2):.2f} MB")

            # Evaluate baseline before training
            print("\n=== Evaluating Baseline Client Models Before Federated Training ===")
            baseline_losses, baseline_accuracies = self.test_baseline()
            for idx, (loss, acc) in enumerate(zip(baseline_losses, baseline_accuracies)):
                print(f"Baseline Client #{idx} -> Test Loss: {loss:.4f}, Test Acc: {acc:.4f}")

        else:
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
        self, model: nn.Module, train_loader: DataLoader, client_idx: int
    ) -> Dict[str, torch.Tensor]:
        """Train a client model.

        Args:
            model (nn.Module): Client's model.
            train_loader (DataLoader): Client's data loader.
            client_idx (int): Client index.

        Returns:
            Dict[str, torch.Tensor]: client model state_dict.
            float: average loss over all epochs.
        """
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

    def _train_consensus_iteration(self, send_buffer: Dict[int, Dict[str, torch.Tensor]]):
        """
        Perform one iteration of consensus-based training.

        Args:
            send_buffer (Dict[int, Dict[str, torch.Tensor]]): Current models from all clients.
        """
        received_buffers = {}
        # Step 1: Each client sends its model to send_buffer
        for i in range(self.args.num_clients):
            received_buffers[i] = copy.deepcopy(send_buffer[i])

        # Step 2: Each client receives models from its neighbors and updates its model
        x_new = {}
        for i in range(self.args.num_clients):
            # 2.1. Identify all connected neighbors
            all_neighbors = self.get_neighbors(i, self.adjacency_matrix)

            # 2.2. Receive models from all connected neighbors via the send_buffer
            received_models = [received_buffers[j] for j in all_neighbors]

            # 2.3. Aggregate received models with the local model (Averaging)
            x_aggregated = self.aggregate_models(received_models, send_buffer[i])

            # 2.4. Compute gradients based on local data
            data_batch = self.sample_local_data(i)
            gradients = self.compute_gradients(x_aggregated, data_batch, i)

            # 2.5. Update local model with aggregated model and gradients
            # Convert aggregated model to tensor and gradients
            updated_state = {}
            for key in x_aggregated.keys():
                updated_state[key] = x_aggregated[key] - self.args.lr * gradients[key]

            x_new[i] = updated_state

        # Step 3: Update all client models simultaneously
        for i in range(self.args.num_clients):
            self.client_models[i].load_state_dict(x_new[i])

    def get_neighbors(self, worker_id: int, adjacency_matrix: np.ndarray) -> List[int]:
        """Get all neighbors (incoming and outgoing) of a worker.

        Args:
            worker_id (int): ID of the worker.
            adjacency_matrix (np.ndarray): Weighted adjacency matrix.

        Returns:
            List[int]: List of neighbor IDs.
        """
        neighbors = []
        for j in range(len(adjacency_matrix)):
            if adjacency_matrix[worker_id][j] > 0 or adjacency_matrix[j][worker_id] > 0:
                neighbors.append(j)
        return neighbors

    def aggregate_models(self, models: List[Dict[str, torch.Tensor]], local_model: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Aggregate (average) models received from neighbors with the local model.

        Args:
            models (List[Dict[str, torch.Tensor]]): List of received models.
            local_model (Dict[str, torch.Tensor]): Local model.

        Returns:
            Dict[str, torch.Tensor]: Aggregated model.
        """
        total_models = models + [local_model]  # Include the local model in the aggregation
        aggregated_model = {}
        for key in local_model.keys():
            aggregated_model[key] = torch.stack([model[key] for model in total_models], dim=0).mean(dim=0)
        return aggregated_model

    def sample_local_data(self, client_id: int) -> Subset:
        """Sample a batch of data for a client.

        Args:
            client_id (int): Client ID.

        Returns:
            Subset: A subset of the client's data.
        """
        # Placeholder for data sampling logic
        # Here, we'll return the entire dataset for simplicity
        return self.train_set

    def compute_gradients(self, model_state: Dict[str, torch.Tensor], data_batch: Subset, client_id: int) -> Dict[str, torch.Tensor]:
        """Compute gradients for the given model and data batch.

        Args:
            model_state (Dict[str, torch.Tensor]): Aggregated model state.
            data_batch (Subset): Batch of data for the client.
            client_id (int): Client ID.

        Returns:
            Dict[str, torch.Tensor]: Gradients.
        """
        model = create_vgg16_for_cifar10(num_classes=10).to(self.device)
        model.load_state_dict(model_state)
        model.train()
        optimizer = optim.SGD(model.parameters(), lr=self.args.lr, momentum=self.args.momentum)
        criterion = nn.CrossEntropyLoss()

        loader = DataLoader(data_batch, batch_size=self.args.batch_size, shuffle=True, num_workers=2)
        epoch_loss = 0.0
        epoch_correct = 0
        epoch_samples = 0

        for data, target in loader:
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

        # Extract gradients
        gradients = {}
        for name, param in model.named_parameters():
            if param.grad is not None:
                gradients[name] = param.grad.data.clone()
            else:
                gradients[name] = torch.zeros_like(param.data)

        return gradients

    def train(self) -> Dict[str, any]:
        """Train the server model using federated learning or consensus-based approach.

        Returns:
            Dict[str, any]: Collected metrics from training.
        """
        train_losses = []
        self.reached_target_at = None
        metrics = {
            "pruning_policy": self.pruning_policy if self.pruning_policy else "No Pruning",
            "pruning_amount": self.pruning_amount,
            "model_size_before_pruning_MB": self.model_size_before_pruning / (1024 ** 2),
            "model_size_after_pruning_MB": None,
            "rounds": []
        }

        for epoch in range(1, self.args.n_epochs + 1):
            if self.consensus:
                print(f"\n=== Consensus-Based Federated Round {epoch}/{self.args.n_epochs} ===")
                # Step 1: Prepare a buffer to store the models to be shared in this iteration
                send_buffer = {i: copy.deepcopy(self.client_models[i].state_dict()) for i in range(self.args.num_clients)}

                # Step 2: Perform consensus-based model exchange and update
                self._train_consensus_iteration(send_buffer)

                # Optionally, evaluate models after each round
                print("\n=== Evaluating Client Models After Federated Training ===")
                round_losses, round_accuracies = self.test_baseline()
                avg_round_loss = np.mean(round_losses)
                avg_round_acc = np.mean(round_accuracies)

                # Measure model size after pruning
                current_model_size = get_model_size(self.client_models[0])
                print(f"Model size after pruning in round {epoch}: {current_model_size / (1024 ** 2):.2f} MB")

                # Update average loss of this round
                train_losses.append(avg_round_loss)

                # Collect round metrics
                round_metrics = {
                    "round": epoch,
                    "avg_train_loss": avg_round_loss,
                    "test_loss": avg_round_loss,  # Placeholder, can be replaced with actual test loss
                    "test_accuracy": avg_round_acc,  # Placeholder, can be replaced with actual test accuracy
                    "model_size_MB": current_model_size / (1024 ** 2)
                }
                metrics["rounds"].append(round_metrics)

                # Print results to CLI
                print(f"\n\nResults after {epoch} rounds of training:")
                print(f"---> Avg Training Loss: {avg_round_loss:.4f}")
                print(f"---> Avg Test Loss: {avg_round_loss:.4f} | Avg Test Accuracy: {avg_round_acc:.4f}\n")

                # Early stopping
                if self.args.early_stopping and avg_round_acc >= self.args.target_acc and self.reached_target_at is None:
                    self.reached_target_at = epoch
                    round_metrics["reached_target_at"] = self.reached_target_at
                    print(
                        f"\n -----> Target accuracy {self.args.target_acc} reached at round {epoch}! <----- \n"
                    )
                    print(f"\nEarly stopping at round #{epoch}...")
                    break

            else:
                print(f"\n=== Federated Round {epoch}/{self.args.n_epochs} ===")
                clients_models = []
                clients_losses = []

                # Train all clients
                for client_idx in range(self.args.num_clients):
                    print(f"\n--- Training on Client {client_idx+1}/{self.args.num_clients} ---")
                    client_model_state, client_loss = self._train_client(
                        model=copy.deepcopy(self.global_model),
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

        if self.consensus:
            # After consensus-based training, measure model size after pruning
            self.model_size_after_pruning = get_model_size(self.client_models[0])
            metrics["model_size_after_pruning_MB"] = self.model_size_after_pruning / (1024 ** 2)
        else:
            # After PS-based training, measure model size after pruning
            self.model_size_after_pruning = get_model_size(self.global_model)
            metrics["model_size_after_pruning_MB"] = self.model_size_after_pruning / (1024 ** 2)

        return metrics

    def test_baseline(self) -> (List[float], List[float]):
        """Test the client models.

        Returns:
            List[float]: List of average losses per client.
            List[float]: List of average accuracies per client.
        """
        self.global_model.eval()

        client_losses = []
        client_accuracies = []

        for idx, model in enumerate(self.client_models):
            model.eval()
            total_loss = 0.0
            total_correct = 0
            total_samples = 0
            criterion = nn.CrossEntropyLoss()

            with torch.no_grad():
                for data, target in self.test_loader:
                    data, target = data.to(self.device), target.to(self.device)
                    outputs = model(data)
                    loss = criterion(outputs, target)

                    total_loss += loss.item() * data.size(0)
                    _, predicted = outputs.max(1)
                    total_correct += predicted.eq(target).sum().item()
                    total_samples += data.size(0)

            avg_loss = total_loss / total_samples
            avg_acc = total_correct / total_samples

            client_losses.append(avg_loss)
            client_accuracies.append(avg_acc)

        return client_losses, client_accuracies

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
    parser.add_argument('--consensus', action='store_true', help='Enable consensus-based message exchange')
    parser.add_argument('--adjacency_matrix_path', type=str, default="adjacency_matrix.npy", help='Path to the adjacency matrix file for consensus-based training')
    parser.add_argument('--noniid', action='store_true', help='Enable non-IID data partitioning among clients')

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Validate arguments for consensus-based approach
    if args.consensus:
        if not os.path.exists(args.adjacency_matrix_path):
            raise FileNotFoundError(f"{args.adjacency_matrix_path} not found. Please provide a valid adjacency matrix for consensus-based training.")
        adjacency_matrix = np.load(args.adjacency_matrix_path)
        print(f"Loaded adjacency matrix from {args.adjacency_matrix_path}")
    else:
        adjacency_matrix = None

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
            pruning_amount=pruning_amount,
            consensus=args.consensus,
            adjacency_matrix=adjacency_matrix
        )

        # Federated training with or without pruning
        if args.consensus:
            print(f"\n=== Consensus-Based Federated Learning Begins: Policy={pruning_policy}, Amount={pruning_amount}, Rounds={args.n_epochs} ===")
        else:
            print(f"\n=== Federated Learning Begins: Policy={pruning_policy}, Amount={pruning_amount}, Rounds={args.n_epochs} ===")
        
        metrics = trainer.train()

        if args.consensus:
            # Evaluate the federated (aggregated) model
            fed_loss, fed_acc = 0.0, 0.0  # Placeholder as consensus-based evaluation is per client
            print(f"\nFinal Federated models have been trained using consensus-based approach.")
        else:
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
