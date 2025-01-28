import numpy as np
import networkx as nx
import os

def generate_adjacency_matrix(num_workers, density, path):
    """
    Generates an adjacency matrix for a given number of workers and density.
    
    Parameters:
    - num_workers (int): Number of worker nodes.
    - density (float): Probability for edge creation (0 <= density <= 1).
                       If density is 0, a ring topology is generated.
    - path (str): File path to save the adjacency matrix (as a .npy file).
    
    Raises:
    - ValueError: If num_workers is less than 2, or density is out of bounds.
    - IOError: If the adjacency matrix cannot be saved to the specified path.
    """
    
    if num_workers < 2:
        raise ValueError("Number of workers must be at least 2 to form a meaningful topology.")
    
    if not (0 <= density <= 1):
        raise ValueError("Density must be between 0 and 1.")
    
    if density == 0:
        # Create a ring (cycle) topology
        G = nx.cycle_graph(num_workers)
        print("Generated a ring (cycle) topology.")
    else:
        # Generate a random connected graph using the Erdős-Rényi model
        # Repeat until a connected graph is obtained
        attempts = 0
        while True:
            G = nx.erdos_renyi_graph(n=num_workers, p=density)
            attempts += 1
            if nx.is_connected(G):
                print(f"Generated a connected random graph after {attempts} attempt(s).")
                break
            else:
                print(f"Attempt {attempts}: Generated graph is not connected. Retrying...")
    
    # Convert the graph to an adjacency matrix (binary: 1 for edge, 0 for no edge)
    adjacency_matrix = nx.to_numpy_array(G, dtype=int)
    
    # Ensure the adjacency matrix is symmetric
    adjacency_matrix = np.maximum(adjacency_matrix, adjacency_matrix.T)
    
    # Save the adjacency matrix to the specified path
    try:
        np.save(path, adjacency_matrix)
        print(f"Adjacency matrix saved to '{path}'.")
    except IOError as e:
        raise IOError(f"Failed to save adjacency matrix to '{path}': {e}")

# Example Usage
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate an adjacency matrix for consensus-based training.")
    parser.add_argument('--num_workers', type=int, required=True, help='Number of worker nodes.')
    parser.add_argument('--density', type=float, required=True, help='Density of the graph (0 for ring topology).')
    parser.add_argument('--output_path', type=str, required=True, help='Path to save the adjacency matrix (.npy file).')

    args = parser.parse_args()

    # Validate the output path's directory exists
    output_dir = os.path.dirname(args.output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        print(f"Created directory '{output_dir}' for adjacency matrix.")

    generate_adjacency_matrix(num_workers=args.num_workers, density=args.density, path=args.output_path)
