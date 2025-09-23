import torch
import time
import matplotlib.pyplot as plt
import numpy as np
from tabulate import tabulate
import seaborn as sns
from datetime import datetime
import os

# Set style for better looking plots
plt.style.use("seaborn-v0_8")
sns.set_palette("husl")


def generate(model, tokenizer, prompt, max_new_tokens=128):
    tokens = tokenizer.encode(prompt)
    tokens = (
        torch.tensor(tokens).unsqueeze(0).to(device)
    )  # unsqueeze to include the batch dimension

    start_pos = tokens.shape[1]
    total_len = start_pos + max_new_tokens
    st = time.monotonic()
    for _ in range(max_new_tokens):
        with torch.no_grad():
            logits = model(tokens)

        next_token = torch.argmax(logits[:, -1, :], dim=-1).unsqueeze(0)
        tokens = torch.cat((tokens, next_token), dim=1)
    et = time.monotonic() - st
    decoded = tokenizer.decode(tokens.squeeze(0).tolist())
    return decoded, et


def benchmark_generation(
    model, tokenizer, prompt, warmup=1, iters=5, max_new_tokens=20, model_name="Model"
):
    print(f"🔥 Warming up {model_name}...")
    for _ in range(warmup):
        _, _ = generate(model, tokenizer, prompt, max_new_tokens=32)

    total_latency = 0
    total_token_sec = 0
    latencies = []
    token_rates = []

    print(f"⚡ Benchmarking {model_name}...")
    for i in range(iters):
        out, latency = generate(model, tokenizer, prompt, max_new_tokens=max_new_tokens)

        in_len = len(tokenizer.encode(prompt))
        gen_len = len(tokenizer.encode(out)) - in_len
        token_rate = gen_len / latency

        total_latency += latency
        total_token_sec += token_rate
        latencies.append(latency)
        token_rates.append(token_rate)

        print(f"  📊 iter {i+1:2d}: latency={latency:.3f}s, tokens/s={token_rate:.2f}")

    avg_latency = total_latency / iters
    avg_token_sec = total_token_sec / iters
    std_latency = np.std(latencies)
    std_token_sec = np.std(token_rates)

    print(
        f"  📈 Tokens: {max_new_tokens} | Avg Latency: {avg_latency:.3f}±{std_latency:.3f}s | Avg Tokens/s: {avg_token_sec:.2f}±{std_token_sec:.2f}"
    )
    print()

    return {
        "avg_latency": avg_latency,
        "avg_token_sec": avg_token_sec,
        "std_latency": std_latency,
        "std_token_sec": std_token_sec,
        "latencies": latencies,
        "token_rates": token_rates,
    }


def print_beautiful_header():
    print("=" * 80)
    print("🚀 QWEN MODEL PERFORMANCE BENCHMARK")
    print("=" * 80)
    print(f"📅 Benchmark Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"🔧 Device: {device}")
    print(f"🧠 Models: Qwen3 (Torch) vs FastQwen3 (CUDA)")
    print("=" * 80)
    print()


def print_comparison_table(results_torch, results_fast, token_sizes):
    """Print a beautiful comparison table"""
    print("📊 PERFORMANCE COMPARISON TABLE")
    print("-" * 100)

    headers = [
        "Tokens",
        "Torch Latency (s)",
        "CUDA Latency (s)",
        "Speedup",
        "Torch Tokens/s",
        "CUDA Tokens/s",
        "Throughput Gain",
    ]

    table_data = []
    for i, tokens in enumerate(token_sizes):
        torch_lat = results_torch[i]["avg_latency"]
        cuda_lat = results_fast[i]["avg_latency"]
        speedup = torch_lat / cuda_lat

        torch_tps = results_torch[i]["avg_token_sec"]
        cuda_tps = results_fast[i]["avg_token_sec"]
        throughput_gain = cuda_tps / torch_tps

        table_data.append(
            [
                tokens,
                f"{torch_lat:.3f} ± {results_torch[i]['std_latency']:.3f}",
                f"{cuda_lat:.3f} ± {results_fast[i]['std_latency']:.3f}",
                f"{speedup:.2f}x",
                f"{torch_tps:.1f} ± {results_torch[i]['std_token_sec']:.1f}",
                f"{cuda_tps:.1f} ± {results_fast[i]['std_token_sec']:.1f}",
                f"{throughput_gain:.2f}x",
            ]
        )

    print(tabulate(table_data, headers=headers, tablefmt="fancy_grid"))
    print()


def create_performance_plots(results_torch, results_fast, token_sizes):
    """Create comprehensive performance visualization"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle("🚀 Qwen Model Performance Comparison", fontsize=16, fontweight="bold")

    # Extract data
    torch_latencies = [r["avg_latency"] for r in results_torch]
    cuda_latencies = [r["avg_latency"] for r in results_fast]
    torch_tps = [r["avg_token_sec"] for r in results_torch]
    cuda_tps = [r["avg_token_sec"] for r in results_fast]

    torch_lat_std = [r["std_latency"] for r in results_torch]
    cuda_lat_std = [r["std_latency"] for r in results_fast]
    torch_tps_std = [r["std_token_sec"] for r in results_torch]
    cuda_tps_std = [r["std_token_sec"] for r in results_fast]

    # Plot 1: Latency Comparison
    axes[0, 0].errorbar(
        token_sizes,
        torch_latencies,
        yerr=torch_lat_std,
        marker="o",
        linewidth=2,
        capsize=5,
        label="Torch (BF16)",
        color="#FF6B6B",
    )
    axes[0, 0].errorbar(
        token_sizes,
        cuda_latencies,
        yerr=cuda_lat_std,
        marker="s",
        linewidth=2,
        capsize=5,
        label="CUDA (FP16)",
        color="#4ECDC4",
    )
    axes[0, 0].set_xlabel("Number of Tokens")
    axes[0, 0].set_ylabel("Latency (seconds)")
    axes[0, 0].set_title("⏱️ Latency Comparison")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_xscale("log")
    axes[0, 0].set_yscale("log")

    # Plot 2: Throughput Comparison
    axes[0, 1].errorbar(
        token_sizes,
        torch_tps,
        yerr=torch_tps_std,
        marker="o",
        linewidth=2,
        capsize=5,
        label="Torch (BF16)",
        color="#FF6B6B",
    )
    axes[0, 1].errorbar(
        token_sizes,
        cuda_tps,
        yerr=cuda_tps_std,
        marker="s",
        linewidth=2,
        capsize=5,
        label="CUDA (FP16)",
        color="#4ECDC4",
    )
    axes[0, 1].set_xlabel("Number of Tokens")
    axes[0, 1].set_ylabel("Tokens per Second")
    axes[0, 1].set_title("🏃 Throughput Comparison")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_xscale("log")

    # Plot 3: Speedup Factor
    speedup = [t / c for t, c in zip(torch_latencies, cuda_latencies)]
    axes[1, 0].plot(
        token_sizes,
        speedup,
        marker="D",
        linewidth=3,
        markersize=8,
        color="#45B7D1",
        label="Speedup Factor",
    )
    axes[1, 0].axhline(y=1, color="red", linestyle="--", alpha=0.7, label="No Speedup")
    axes[1, 0].set_xlabel("Number of Tokens")
    axes[1, 0].set_ylabel("Speedup Factor")
    axes[1, 0].set_title("⚡ Latency Speedup (CUDA vs Torch)")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_xscale("log")

    # Add speedup values as annotations
    for i, (x, y) in enumerate(zip(token_sizes, speedup)):
        axes[1, 0].annotate(
            f"{y:.1f}x",
            (x, y),
            textcoords="offset points",
            xytext=(0, 10),
            ha="center",
            fontweight="bold",
        )

    # Plot 4: Throughput Gain
    throughput_gain = [c / t for t, c in zip(torch_tps, cuda_tps)]
    axes[1, 1].plot(
        token_sizes,
        throughput_gain,
        marker="*",
        linewidth=3,
        markersize=10,
        color="#96CEB4",
        label="Throughput Gain",
    )
    axes[1, 1].axhline(y=1, color="red", linestyle="--", alpha=0.7, label="No Gain")
    axes[1, 1].set_xlabel("Number of Tokens")
    axes[1, 1].set_ylabel("Throughput Gain Factor")
    axes[1, 1].set_title("🎯 Throughput Gain (CUDA vs Torch)")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_xscale("log")

    # Add gain values as annotations
    for i, (x, y) in enumerate(zip(token_sizes, throughput_gain)):
        axes[1, 1].annotate(
            f"{y:.1f}x",
            (x, y),
            textcoords="offset points",
            xytext=(0, 10),
            ha="center",
            fontweight="bold",
        )

    plt.tight_layout()

    # Save the plot
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"qwen_benchmark_{timestamp}.png"
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    print(f"📈 Performance plots saved as: {filename}")
    plt.show()


def print_summary_stats(results_torch, results_fast, token_sizes):
    """Print summary statistics"""
    print("📋 SUMMARY STATISTICS")
    print("-" * 50)

    # Calculate average speedup
    speedups = []
    throughput_gains = []

    for i in range(len(token_sizes)):
        speedup = results_torch[i]["avg_latency"] / results_fast[i]["avg_latency"]
        t_gain = results_fast[i]["avg_token_sec"] / results_torch[i]["avg_token_sec"]
        speedups.append(speedup)
        throughput_gains.append(t_gain)

    avg_speedup = np.mean(speedups)
    max_speedup = np.max(speedups)
    min_speedup = np.min(speedups)

    avg_t_gain = np.mean(throughput_gains)
    max_t_gain = np.max(throughput_gains)
    min_t_gain = np.min(throughput_gains)

    print(f"🏆 Average Speedup: {avg_speedup:.2f}x")
    print(
        f"⚡ Maximum Speedup: {max_speedup:.2f}x (at {token_sizes[np.argmax(speedups)]} tokens)"
    )
    print(
        f"🐌 Minimum Speedup: {min_speedup:.2f}x (at {token_sizes[np.argmin(speedups)]} tokens)"
    )
    print()
    print(f"📊 Average Throughput Gain: {avg_t_gain:.2f}x")
    print(
        f"🚀 Maximum Throughput Gain: {max_t_gain:.2f}x (at {token_sizes[np.argmax(throughput_gains)]} tokens)"
    )
    print(
        f"📉 Minimum Throughput Gain: {min_t_gain:.2f}x (at {token_sizes[np.argmin(throughput_gains)]} tokens)"
    )
    print()


if __name__ == "__main__":
    from llm.qwen3.qwen_torch import Qwen3
    from llm.qwen3.fast_qwen_cuda import FastQwen3
    from llm.qwen3.qwen_token import Qwen3Tokenizer
    from llm.qwen3.config import QwenConfig_bfloat16, QwenConfig_float16

    device = torch.device("cuda")
    tokenizer_file_path = "/home/aman/code/model_go_brr/Qwen3-0.6B/tokenizer.json"

    # Print beautiful header
    print_beautiful_header()

    # Initialize configurations and tokenizer
    config_qwen_bf16 = QwenConfig_bfloat16()
    config_qwen_fp16 = QwenConfig_float16()
    tokenizer = Qwen3Tokenizer(
        tokenizer_file_path=tokenizer_file_path,
        add_gen_prompt=True,
        add_thinking=True,
    )

    # Initialize models
    print("🔄 Loading models...")
    model_torch = Qwen3(config_qwen_bf16).to(device)
    model_fast = FastQwen3(config_qwen_fp16).to(device)
    print("✅ Models loaded successfully!")
    print()

    # Test prompt
    prompt = "Explain the concept of artificial intelligence and its applications in modern technology"

    max_new_token_range = [100 , 200 , 300 , 400 , 500 , 600 , 700 , 800]

    results_torch = []
    results_fast = []

    print("🎯 Starting comprehensive benchmark...")
    print()

    for token_size in max_new_token_range:
        print(f"🧪 Testing with {token_size} tokens...")
        print("-" * 40)

        # Benchmark Torch model
        result_torch = benchmark_generation(
            model_torch,
            tokenizer,
            prompt,
            warmup=2,
            iters=5,
            max_new_tokens=token_size,
            model_name="Qwen3-Torch",
        )
        results_torch.append(result_torch)

        # Benchmark Fast CUDA model
        result_fast = benchmark_generation(
            model_fast,
            tokenizer,
            prompt,
            warmup=2,
            iters=5,
            max_new_tokens=token_size,
            model_name="FastQwen3-CUDA",
        )
        results_fast.append(result_fast)

        # Quick comparison for this token size
        speedup = result_torch["avg_latency"] / result_fast["avg_latency"]
        throughput_gain = result_fast["avg_token_sec"] / result_torch["avg_token_sec"]

        print(f"💫 Quick Stats for {token_size} tokens:")
        print(f"   ⚡ Speedup: {speedup:.2f}x")
        print(f"   📈 Throughput Gain: {throughput_gain:.2f}x")
        print()

    # Print comprehensive results
    print("🎉 BENCHMARK COMPLETE!")
    print("=" * 80)
    print()

    print_comparison_table(results_torch, results_fast, max_new_token_range)
    print_summary_stats(results_torch, results_fast, max_new_token_range)

    # Create and show plots
    create_performance_plots(results_torch, results_fast, max_new_token_range)

    print("=" * 80)
    print(
        "✨ Benchmark analysis complete! Check the generated plots for detailed insights."
    )
    print("=" * 80)
