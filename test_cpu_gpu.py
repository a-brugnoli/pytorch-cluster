#!/usr/bin/env python3
"""Learn free fall with an MLP; benchmark CPU and GPU training steps."""
import argparse
import copy
import json
import os
import platform
import statistics
import time
from pathlib import Path

import torch
from torch import nn


def synchronize(device):
    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=4096)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--threads", type=int,
                        default=int(os.environ.get("SLURM_CPUS_PER_TASK", "4")))
    parser.add_argument("--cpu-only", action="store_true")
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("results.json"))
    args = parser.parse_args()
    for name in ("samples", "width", "steps", "repeats", "threads"):
        if getattr(args, name) < 1:
            parser.error(f"--{name} must be positive")
    if args.warmup < 0:
        parser.error("--warmup must be nonnegative")
    if args.cpu_only and args.require_cuda:
        parser.error("--cpu-only and --require-cuda cannot be combined")
    if args.require_cuda and not torch.cuda.is_available():
        raise SystemExit("CUDA unavailable. Check the GPU allocation and PyTorch environment.")

    torch.set_num_threads(args.threads)
    torch.manual_seed(42)
    # t in seconds; d = 0.5 * 9.81 * t**2 in meters.
    t_max = 2.0
    gravity = 9.81
    d_max = 0.5 * gravity * t_max**2


    t = t_max * torch.rand(args.samples, 1)
    x_cpu = t / t_max
    y_cpu = (0.5 * gravity * t.square()) / d_max
    initial_model = nn.Sequential(
        nn.Linear(1, args.width), nn.ReLU(),
        nn.Linear(args.width, args.width), nn.ReLU(),
        nn.Linear(args.width, 1),
    )

    def benchmark(device):
        x, y = x_cpu.to(device), y_cpu.to(device)
        times, errors = [], []
        for trial in range(args.repeats):
            model = copy.deepcopy(initial_model).to(device)
            # Explicitly use the same optimizer implementation on both devices.
            optimizer = torch.optim.Adam(model.parameters(), lr=0.003, foreach=False)

            def train_step():
                optimizer.zero_grad(set_to_none=True)
                loss = (model(x) - y).square().mean()
                loss.backward()
                optimizer.step()

            for _ in range(args.warmup):
                train_step()
            synchronize(device)
            start = time.perf_counter()
            for _ in range(args.steps):
                train_step()
            synchronize(device)
            elapsed = time.perf_counter() - start
            times.append(elapsed)

            model.eval()
            with torch.no_grad():
                test_t = torch.linspace(0, 2, 1000, device=device).reshape(-1, 1)
                predicted = model(test_t / 2.0) * 19.62
                expected = 0.5 * 9.81 * test_t.square()
                rmse = (predicted - expected).square().mean().sqrt().item()
            errors.append(rmse)
            print(f"{device.upper()} trial {trial + 1}: {elapsed:.3f} s; "
                  f"RMSE {rmse:.4f} m", flush=True)
        median = statistics.median(times)
        return {"seconds": times, "median_seconds": median,
                "milliseconds_per_step": 1000 * median / args.steps,
                "rmse_meters": errors}

    gpu = None
    if not args.cpu_only:
        if torch.cuda.is_available():
            gpu = "cuda"
        elif torch.backends.mps.is_available():
            gpu = "mps"
    config = vars(args).copy()
    config["output"] = str(args.output)
    results = {
        "host": platform.node(), "platform": platform.platform(),
        "pytorch_version": str(torch.__version__), "cuda_version": torch.version.cuda,
        "cpu_threads": torch.get_num_threads(), "config": config,
        "gpu": (torch.cuda.get_device_name(0) if gpu == "cuda" else gpu),
        "timing_scope": "training steps only; setup, transfers, warmup and evaluation excluded",
    }
    print(f"Host: {results['host']}; PyTorch: {torch.__version__}; "
          f"CPU threads: {results['cpu_threads']}; GPU: {results['gpu']}", flush=True)
    results["cpu"] = benchmark("cpu")
    if gpu:
        results[gpu] = benchmark(gpu)
        ratio = results["cpu"]["median_seconds"] / results[gpu]["median_seconds"]
        results["gpu_speedup"] = ratio
        results["training_time_reduction_percent"] = 100 * (1 - 1 / ratio)
        print(f"GPU speedup: {ratio:.2f}x; training time reduction: "
              f"{results['training_time_reduction_percent']:.1f}%", flush=True)
    else:
        print("CPU-only run: no GPU comparison.", flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {args.output}", flush=True)


if __name__ == "__main__":
    main()