# PyTorch on a Slurm cluster

Verify a GPU-enabled PyTorch installation and compare CPU and GPU training times with a small neural network.

The working cluster setup uses **Python 3.11, PyTorch 2.5.1 with CUDA 11.8, and NumPy 1.26.4**. Conda manages the environment; PyTorch and NumPy are installed from wheels with pip.

## Repository files

| File | Purpose |
| --- | --- |
| [verify_torch_installation.sh](verify_torch_installation.sh) | Print CUDA information and perform a matrix multiplication on the GPU. |
| [run_verification.slurm](run_verification.slurm) | Submit the installation verification to a GPU node. |
| [test_cpu_gpu.py](test_cpu_gpu.py) | Train the same model on CPU and GPU, compare timings, and save JSON results. |
| [run_benchmark.slurm](run_benchmark.slurm) | Allocate one GPU and four CPU cores, then run the benchmark. |

## 1. Log in to the cluster

Start on your **own computer**, using Terminal on Linux/macOS or PowerShell with OpenSSH on Windows. You need a cluster account and the login address supplied by your administrators. Connect to the institution's VPN first if remote access requires it.

Replace `YOUR_USERNAME` and `login.cluster.example.org` with your cluster username and the actual login hostname. The hostname below is a placeholder:

```bash
ssh YOUR_USERNAME@login.cluster.example.org
```

For example, a username such as `a.brugnoli` goes before `@`; the address after `@` must be the cluster's real login endpoint. A name displayed in a remote prompt, such as `pando01`, may be an internal hostname rather than an address reachable from your computer.

Follow the cluster's password or multi-factor authentication prompts. Password characters are normally hidden while you type. On the first connection, compare the displayed host-key fingerprint with the one published by the cluster administrators before accepting it.

### If your account uses an SSH key

The private key must be available on the computer from which you connect, and its matching public key must already be registered for your cluster account using the institution's approved procedure. To select an existing private key explicitly:

```bash
ssh -i ~/.ssh/id_ed25519 YOUR_USERNAME@login.cluster.example.org
```

Use your actual key path if it differs. Enter the key's passphrase if prompted. Keep the private key private; only the `.pub` file is used for registration.

Cluster login and GitHub authentication are separate. Adding a key to GitHub enables GitHub access; it does not authorize login to the cluster. Likewise, a key stored on the cluster is not automatically available on your laptop.

### If the cluster requires a gateway

Only use this form if the administrators provide a gateway or jump host:

```bash
ssh -J GATEWAY_USERNAME@gateway.example.org YOUR_USERNAME@login.cluster.example.org
```

Replace both hostnames and both usernames with the supplied connection details. The gateway and login node may use different credentials. See the [OpenSSH manual](https://man.openbsd.org/ssh) for the `-i` and `-J` options.

### Confirm where you are

After login, run:

```bash
hostname
whoami
pwd
```

These commands show the remote hostname, your cluster username, and your current directory. Continue with the repository and environment setup below **in this remote shell**. Files and environments on your laptop are separate from those on the cluster.

Use the login node for setup, editing, and submitting jobs. Submit GPU verification and training through Slurm; the scheduler chooses a compute node. To disconnect when finished:

```bash
exit
```

Jobs already submitted with `sbatch` continue after you disconnect.

## 2. Get the code on the cluster

In the remote shell, move to the project directory recommended by your cluster administrators, then clone the repository:

```bash
git clone https://github.com/a-brugnoli/pytorch-cluster.git
cd pytorch-cluster
```

Submit jobs from this directory: the Slurm scripts locate the Python and shell files relative to the submission directory.

This public repository can be cloned over HTTPS without registering a GitHub SSH key. If you already cloned it on the cluster, enter the existing directory instead of cloning it again.

## 3. Create the cluster environment

These instructions describe the tested Linux x86-64 cluster setup. They require Conda and a compatible NVIDIA driver supplied by the cluster.

On this cluster, Conda is available through:

```bash
module load python/2023-3
eval "$(conda shell.bash hook)"
```

Use the module name provided by your administrators on another cluster. If Conda is already available and initialized, the module step is unnecessary.

Create the environment once:

```bash
conda create -n pytorch-wheel python=3.11 pip -y
conda activate pytorch-wheel

python -m pip install torch==2.5.1 \
    --index-url https://download.pytorch.org/whl/cu118

python -m pip install --only-binary=:all: "numpy==1.26.4"
```

The pinned versions avoid the library and compiler compatibility issues encountered on this older cluster. The CUDA 11.8 runtime is supplied through the PyTorch installation; ordinary use does not require loading a separate CUDA toolkit module.

Check imports:

```bash
python -c "import numpy, torch; print('NumPy:', numpy.__version__); print('PyTorch:', torch.__version__); print('CUDA runtime:', torch.version.cuda)"
```

Expected versions:

```text
NumPy: 1.26.4
PyTorch: 2.5.1+cu118
CUDA runtime: 11.8
```

Importing PyTorch can be checked on a login node. GPU access and computation must be checked inside a GPU allocation.

## 4. Adapt the Slurm settings

Both Slurm scripts use:

```bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4

module load python/2023-3
eval "$(conda shell.bash hook)"
conda activate pytorch-wheel
```

Adjust the partition, GPU request, module, time limit, and any required account or QoS settings to match your cluster.

A generic GPU request can allocate any eligible GPU model. To test a V100 specifically, use the V100 partition, GPU type, or constraint configured by your administrators. The example results below were measured on an A100.

The benchmark additionally requests 8 GB of host memory and a 20-minute time limit. It uses one GPU; it does not perform distributed or multi-GPU training.

## 5. Verify the installation

Submit the verification job:

```bash
sbatch run_verification.slurm
```

Slurm prints the job ID. Check its status:

```bash
squeue -u "$USER"
```

Replace `JOB_ID` below with the number returned by Slurm:

```bash
cat slurm.JOB_ID.out
cat slurm.JOB_ID.err
```

A successful run includes:

```text
PyTorch: 2.5.1+cu118
CUDA runtime: 11.8
CUDA available: True
GPU: NVIDIA A100-PCIE-40GB
Compute capability: (8, 0)
GPU computation successful
```

The GPU name and compute capability depend on the allocation. A V100 has compute capability `(7, 0)`.

## 6. Benchmark CPU versus GPU

Submit the benchmark:

```bash
sbatch run_benchmark.slurm
```

The CPU and CUDA runs execute sequentially on the same allocated node. CUDA is required by the launcher; the job exits with an error if it is unavailable.

| Parameter | Default |
| --- | ---: |
| Training samples | 4096 |
| Hidden-layer width | 128 |
| Timed training steps per trial | 300 |
| Warmup steps per trial | 20 |
| Repetitions per device | 3 |
| CPU threads | 4, from the Slurm CPU allocation |

Arguments after the Slurm filename override benchmark defaults. For a larger workload:

```bash
sbatch run_benchmark.slurm --samples 16384 --width 512
```

For more measurements:

```bash
sbatch run_benchmark.slurm --steps 1000 --repeats 5
```

To change the CPU allocation, keep the benchmark thread count aligned with it:

```bash
sbatch --cpus-per-task=8 run_benchmark.slurm
```

Slurm options go **before** the script filename; benchmark arguments go **after** it.

### Output files

| File | Contents |
| --- | --- |
| `benchmark.JOB_ID.out` | Environment information, per-trial timings, RMSE, and GPU speedup. |
| `benchmark.JOB_ID.err` | Errors and warnings. |
| `results/benchmark-JOB_ID.json` | Configuration, hardware information, timings, RMSE, and speedup. |

The results directory is created automatically. For an example job ID:

```bash
cat benchmark.4023303.out
cat benchmark.4023303.err
python -m json.tool results/benchmark-4023303.json
```

### What is measured?

The model is a multilayer perceptron with two hidden layers and ReLU activations. It learns free-fall distance:

```text
d(t) = 0.5 × 9.81 × t², for t between 0 and 2 seconds
```

Inputs and targets are normalized. Training uses full-batch mean squared error and Adam with `foreach=False` on both devices. Every trial starts from a copy of the same initial model and uses the same training data.

Each trial performs warmup steps, then times forward passes, backpropagation, and optimizer updates. CUDA synchronization before and after timing ensures that GPU work completes within the measured interval.

**Timing excludes model and optimizer setup, data transfers, warmup, evaluation, environment startup, and queue time.** It measures training throughput for this workload rather than total job duration.

The script reports the median across repetitions:

```text
GPU speedup = median CPU time / median GPU time
Training time reduction (%) = 100 × (1 − median GPU time / median CPU time)
```

A speedup above 1 means the GPU was faster. Small models can be limited by GPU launch overhead, so larger sample counts or layer widths may produce different speedups.

### Example measured result

A run on **23 October 2026**, using an **NVIDIA A100-PCIE-40GB**, four CPU threads, and the default benchmark settings produced:

| Device | Trial 1 | Trial 2 | Trial 3 | Median | RMSE per trial |
| --- | ---: | ---: | ---: | ---: | ---: |
| CPU, 4 threads | 1.595 s | 1.147 s | 1.179 s | 1.179 s | 0.0337 m |
| A100 GPU | 0.354 s | 0.353 s | 0.349 s | 0.353 s | 0.0337 m |

Reported GPU speedup: **3.34×**. Reported training time reduction: **70.0%**.

The prediction error was approximately 3.37 cm on both devices at the printed precision. Timings are illustrative: CPU hardware, GPU model, workload size, and contention affect the result. These measurements do not establish V100 performance.

## Run without Slurm

On a workstation or inside an existing allocation, activate an appropriate PyTorch environment and run:

```bash
python test_cpu_gpu.py --output results/local.json
```

The script selects CUDA when available, otherwise Apple MPS when available, and otherwise runs on CPU only. The Linux CUDA installation instructions above are specific to the cluster; use the appropriate PyTorch package for another operating system.

Force a CPU-only benchmark:

```bash
python test_cpu_gpu.py --cpu-only --threads 4 --output results/cpu-only.json
```

Require an NVIDIA GPU:

```bash
python test_cpu_gpu.py --require-cuda --output results/cuda.json
```

List all options:

```bash
python test_cpu_gpu.py --help
```

On a managed cluster, run compute workloads on allocated compute nodes.

## Troubleshooting

| Symptom | What to check |
| --- | --- |
| Cluster SSH connection times out or the hostname cannot be resolved | Check the supplied login address, network access, and any required VPN or gateway. |
| Cluster SSH reports `Permission denied (publickey)` | Check the cluster username and selected local key. Its public key must be registered with the cluster, not just GitHub. Follow the administrators' authentication procedure. |
| `GLIBC_2.27 not found` involving `libcurand.so.10` | Confirm that the job activates `pytorch-wheel`. This failure occurred with the earlier Conda-installed CUDA library; the fresh environment and official CUDA 11.8 wheel resolved it on the tested cluster. |
| `No module named 'numpy'` | Activate `pytorch-wheel` and install `numpy==1.26.4` with the binary-only command above. |
| NumPy attempts compilation and requires GCC ≥9.3 | Use `python -m pip install --only-binary=:all: "numpy==1.26.4"` to select the compatible wheel. |
| `CUDA unavailable` or `CUDA available: False` | Check that the job has a GPU allocation and uses the intended environment. Inspect `nvidia-smi` and the error log inside that allocation. |
| `conda: command not found` or activation fails | Load the cluster's Conda module and initialize the Bash hook before activation. |
| Missing Python or shell script | Run `sbatch` from the repository root, where the scripts are located. |
| Missing `.log` file | These launchers write `.out` and `.err` files. Use the filenames listed above. |
| `bash: CPU: command not found` after viewing results | Output text was pasted into the shell. Only enter commands from Bash code blocks; the completed benchmark is unaffected. |

The `CUDA Version` shown by `nvidia-smi` describes the driver's CUDA support. `torch.version.cuda` describes the runtime used by PyTorch. The tested driver reports CUDA 12.4 while PyTorch uses CUDA 11.8; these values do not need to be identical.

For package details, see the official [PyTorch previous-version installation instructions](https://pytorch.org/get-started/previous-versions/) and [NumPy 1.26.4 release files](https://pypi.org/project/numpy/1.26.4/#files).

