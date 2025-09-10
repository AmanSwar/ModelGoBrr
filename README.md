# Model Go Brrr

This repository is a collection of experiments and implementations of custom Triton kernels for accelerating large language models. It's a personal project driven by a passion for making models run as fast as possible. This is not intended to be a production-ready library, but rather a playground for exploring performance optimizations in the world of deep learning.

## Implemented Kernels

This repository contains custom Triton kernels for the following operations:

*   **Grouped-Query Attention (GQA):** An optimized GQA implementation that leverages Triton to fuse operations and reduce memory bandwidth.
*   **Fused Feed-Forward Network (FFN) with SiLU:** A fused FFN that combines the linear transformations and SiLU activation into a single kernel, reducing kernel launch overhead and improving data locality.
*   **RMS Normalization:** A custom Triton implementation of RMS Normalization for improved performance over the standard PyTorch version.
*   **Flash Attention:** A Triton implementation of the Flash Attention algorithm for efficient attention computation.
*   **Fused Linear:** A fused linear layer implementation.

## Model

The primary model used for testing and benchmarking in this repository is **Qwen3-0.6B** (MORE MODEL COMING SOON ....). The `llm/qwen3` directory contains the model definition, loading code, and inference scripts. There are two versions of the model implementation:

*   `qwen_torch.py`: A baseline implementation using standard PyTorch modules.
*   `qwen_fast.py`: An optimized implementation that uses the custom Triton kernels.

## Benchmarking

The repository includes benchmarking scripts to compare the performance of the custom Triton kernels against the baseline PyTorch implementations. You can find these scripts in the `kernels` directory and in the main execution blocks of the model files.

## Getting Started

To run the code in this repository, you will need to have PyTorch and Triton installed. You will also need to download the Qwen3-0.6B model weights.

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/model_go_brr.git
    ```
2.  **Install dependencies:**
    ```bash
    pip install torch triton
    ```
3.  **Download the model weights:**
    Download the Qwen3-0.6B model from the Hugging Face Hub and place it in the `Qwen3-0.6B` directory.
4.  **Run the inference script:**
    ```bash
    python llm/qwen3/qwen_fast.py
    ```

## Disclaimer
This is a personal project and is not intended for production use. The code is provided as-is, without any guarantees of correctness or performance.
