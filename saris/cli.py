import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"  # to avoid memory fragmentation
# os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

# Jax acceleration flags
os.environ["XLA_FLAGS"] = (
    "--xla_gpu_enable_triton_softmax_fusion=true "
    "--xla_gpu_triton_gemm_any=True "
    "--xla_gpu_enable_async_collectives=true "
    "--xla_gpu_enable_latency_hiding_scheduler=true "
    "--xla_gpu_enable_highest_priority_async_stream=true "
)

import argparse


def main():
    args = parse_agrs()


def parse_agrs():
    parser = argparse.ArgumentParser()
    parser.add_argument("--drl_config_file", "-dcfg", type=str, required=True)
    parser.add_argument("--sionna_config_file", "-scfg", type=str, required=True)
    parser.add_argument("--command", "-cmd", type=str, required=True)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--log_interval", type=int, default=1)
    parser.add_argument("--verbose", "-v", action="store_true")

    args = parser.parse_args()
    for k, v in args.__dict__.items():
        print(f"{k}: {v}")
    return args


if __name__ == "__main__":
    main()
