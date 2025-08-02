# High-Performance Embedding Service
A gRPC-based embedding service with dynamic batching for optimal throughput and latency. Supports both text and image embeddings using Nomic AI models.

# Features
Dynamic Batching: Intelligent request batching that balances latency and throughput
Multi-Modal: Text embeddings (nomic-embed-text-v2-moe) and image embeddings (nomic-embed-vision-v1.5)
GPU Acceleration: Automatic device detection (CUDA/MPS/CPU)
gRPC API: High-performance async gRPC interface
Production Ready: Robust error handling, graceful shutdown, flexible port binding

# Quick Start
bashpip install grpcio grpcio-tools torch sentence-transformers transformers pillow aiohttp
python embedding_server.py
License
AGPL-3.0 (AI models retain their original Apache 2.0 licenses)
