# Distributed P2P Accelerated RL Training System

## Overview

This project implements a distributed system for accelerated reinforcement learning (RL) training using peer-to-peer (P2P) technologies and parallel programming. The system leverages the computational power of multiple machines, each potentially equipped with GPUs, to accelerate the training of RL agents through distributed computing and federated learning.

## Key Features

- Distributed RL training across a network of peer nodes
- P2P communication for efficient model updates and task distribution
- GPU acceleration for RL computations
- Hybrid Python-C++ implementation for both ease of use and high performance
- Containerized deployment for consistency and scalability
- Federated learning for collaborative model improvement

## System Architecture

The system consists of the following key components:

1. **Peer Nodes**: Each peer in the network runs an instance of the application, containing:
   - An RL agent implemented in C++ for high-performance training
   - A P2P networking module for communication with other peers
   - A task execution module for running distributed RL tasks

2. **P2P Network**: Peers form a decentralized network for:
   - Sharing model updates
   - Distributing training tasks
   - Aggregating learning results

3. **RL Environment**: Implemented in C++ for fast simulation and parallel execution of multiple environments.

4. **Model Aggregator**: Implements federated learning to combine model updates from multiple peers.

5. **Task Distributor**: Manages the distribution of RL tasks among available peers in the network.

## Technologies Used

- **Python**: High-level coordination, P2P networking, and system management
- **C++**: Performance-critical RL algorithm implementation and environment simulation
- **libtorch**: C++ interface to PyTorch for neural network operations
- **libp2p**: P2P networking library
- **CUDA**: GPU acceleration for RL computations
- **pybind11**: Creating Python bindings for C++ modules
- **Docker**: Containerization for consistent deployment
- **CMake**: C++ project configuration and building

## Parallel Programming and Accelerated Computing

This system leverages parallel programming and accelerated computing in several ways:

1. **GPU Acceleration**: The RL agent and neural network operations use CUDA for GPU acceleration, significantly speeding up computations.

2. **Distributed Computing**: Multiple peer nodes work in parallel on different parts of the RL problem, effectively distributing the computational load.

3. **Parallel Environment Simulation**: The C++ environment implementation allows for parallel simulation of multiple environments, increasing the speed of data generation for training.

4. **Federated Learning**: The system uses federated learning techniques to parallelize the learning process across multiple peers, allowing for efficient use of distributed computational resources.

## Getting Started

### Prerequisites

- CUDA-capable GPU
- Docker and docker-compose
- CMake (version 3.10 or higher)
- Python 3.7 or higher

### Installation

1. Clone the repository:
   ```
   git clone git@github.com:aneangel/distributed-macs.git
   cd distributed-macs
   ```

2. Build the Docker image:
   ```
   docker build -t distributed-rl -f docker/Dockerfile .
   ```

3. Start the application using docker-compose:
   ```
   docker-compose -f docker/docker-compose.yml up
   ```

### Usage

1. Configure the system by editing `configs/config.yaml`.

2. Run multiple instances of the application to create a P2P network:
   ```
   docker-compose -f docker/docker-compose.yml up --scale peer=<number_of_peers>
   ```

3. Monitor the training progress through the logs.

## Contributing

Contributions to improve the system are welcome. Please follow these steps:

1. Fork the repository
2. Create a new branch (`git checkout -b feature/your-feature`)
3. Make your changes
4. Commit your changes (`git commit -am 'Add some feature'`)
5. Push to the branch (`git push origin feature/your-feature`)
6. Create a new Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- OpenAI for their work on reinforcement learning algorithms
- The PyTorch team for libtorch
- The libp2p community for their P2P networking library