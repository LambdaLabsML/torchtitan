# DEBUG_FOR_BASELINE_64xGB300

Everything that had to be debugged, installed or changed to get the
`gpt_oss_20b` baseline running on 64x GB300 (16 nodes x 4) at yqb01-qa01,
2026-09-05/06.

The nodes were **bare metal Ubuntu 24.04 (aarch64)** — no GPU driver, no CUDA, no
Slurm, no shared filesystem, no InfiniBand, no containers. Nothing below was
present at the start.

Read this as: *if you rebuild this cluster from bare metal, these are the traps.*

Cluster-side Ansible lives in `~/cluster-setup` on `yqb01-qa01-mgx-00057`.

---

## TL;DR — the four that actually blocked the run

| # | Symptom | Real cause |
| --- | --- | --- |
| 1 | `nvidia-smi`: "No devices were found" | GB300 needs `memhp_default_state=online_movable` |
| 2 | CUDA works over SSH, `No CUDA GPUs are available` under Slurm | Slurm cpuset masked the GPU NUMA nodes |
| 3 | NCCL: `MNNVL (cliqueSize 64) is available but not working` | IMEX absent — these 16 nodes are one NVLink fabric |
| 4 | `ModuleNotFoundError: No module named 'flash_attn'` | Blackwell wants FA4, which is a *separate* package |

Items 2 and 3 are the dangerous ones: each has a "fix" that makes the error go
away while silently producing a much worse number.

---

## 1. GPU driver — `nvidia-smi` finds no devices

**Symptom.** Driver installs cleanly, DKMS builds, `dmesg` shows all four GPUs
initialising via DRM — and then `nvidia-smi` says *"No devices were found."*

```
NVRM: Failing GPU memory onlining as the onlining zone is not movable. pa: 0x484260000000
NVRM: GPU memory zone movable auto onlining failed!
NVRM: RmInitAdapter failed! (0x25:0x40:1249)
```

**Cause.** GB300 onlines GPU HBM as NUMA system memory, and the kernel will only
do that into `ZONE_MOVABLE`.

**Fix.** `memhp_default_state=online_movable` on the kernel command line, then
reboot. In `/etc/default/grub`, applied by `30-gpu-install.yml`.

**Also required:** the **`-open`** driver variant. Blackwell-class datacenter GPUs
are only supported by the open kernel modules —
`nvidia-driver-580-server-open`, not `nvidia-driver-580-server`.

**Not** required: `nvidia-fabricmanager`. There is no NVSwitch inside these nodes
(`lspci` finds none; the four GPUs are directly NVLink-connected at NV18), so it
exits 1 with `Nothing to do [NV_WARN_NOTHING_TO_DO]` on every boot. It is masked
deliberately. This is unrelated to IMEX (§3) — the NVL72 switch trays live outside
the compute node, which is why they never show up in `lspci`.

---

## 2. CUDA invisible under Slurm — the expensive one

**Symptom.** CUDA works perfectly over SSH. Under `srun`, on the same node, with
`nvidia-smi` listing all four GPUs inside the step and `CUDA_VISIBLE_DEVICES=0,1,2,3`:

```
RuntimeError: No CUDA GPUs are available
```

`torch.cuda.device_count()` returns 4. It is `_cuda_init()` that fails.

**What it is not.** Checked and eliminated: device-node permissions
(`/dev/nvidia*` all open fine inside the step, including `nvidia-uvm`), memlock
limits, and the cgroup memory limit — raising `memory.max` to unlimited changed
nothing.

**Cause.** `dmesg` gives it away:

```
NVRM: Assertion failed: (status == NV_OK) || ((status == NV_ERR_NO_MEMORY) && ...) @ pool_alloc.c:601
NVRM: Assertion failed: status == NV_OK @ vaspace_api.c:781
```

Slurm's cpuset derives `cpuset.mems` from the NUMA nodes backing the allocated
CPUs, which here is `[0-1]`. But the machine exposes
`0-1,3-9,11-17,19-25,27-33` — the extras being coherent GPU memory that the driver
allocates from during init. Confined to `[0-1]`, VA-space allocation fails.

Reproduces with all 144 CPUs allocated, so it is the **mems mask, not CPU count**.

**Fix.** `ConstrainCores=no` in `/etc/slurm/cgroup.conf` (`60-cgroup-fix.yml`).

**Trade-off, stated honestly:** jobs sharing a node are no longer CPU/NUMA
isolated from one another. Fine when benchmark jobs take whole nodes; revisit if
this cluster ever runs several concurrent jobs per node.

`ConstrainDevices=yes` is kept — it works, and it is what makes `--gres=gpu:N`
actually restrict a job to its GPUs.

---

## 3. InfiniBand was entirely absent

**Symptom.** None — and that is the problem. Everything "works", just slowly.

`ls /sys/class/infiniband` → *No such file or directory*, on nodes carrying four
ConnectX-8 HCAs.

**Cause.** Ubuntu's `linux-modules` package does not ship `mlx5_ib`. It lives in
`linux-modules-extra-$(uname -r)`, which was not installed. With no IB device,
NCCL silently falls back to TCP over the 400 GbE.

**Fix** (`50-infiniband.yml`): install `linux-modules-extra-$(uname -r)`,
`rdma-core`, `ibverbs-utils`, `infiniband-diags`; `modprobe mlx5_ib`; and add
`/etc/modules-load.d/mlx5_ib.conf` so it survives a reboot — `mlx5_core`
autoloads by PCI ident but does **not** pull in `mlx5_ib`.

After the fix, every node reports:

```
ibP16p3s0  ACTIVE  800 Gb/sec (4X XDR)
ibP18p3s0  ACTIVE  800 Gb/sec (4X XDR)
ibP2p3s0   ACTIVE  800 Gb/sec (4X XDR)
ibp3s0     ACTIVE  800 Gb/sec (4X XDR)
roceP22p1s0f0 ACTIVE 400 Gb/sec (4X NDR)   <- Ethernet, not for NCCL
```

**Gotcha:** install `rdma-core` *before* first loading `mlx5_ib`. Loading it first
gives the legacy `mlx5_0..4` names instead of the persistent `ibP*` ones, because
rdma-core supplies the udev rules. One node ended up inconsistent this way and had
to have the module reloaded (the ports then take ~10 s to re-train).

`NCCL_IB_HCA` in the launch script names the four IB devices and deliberately
excludes `roceP22p1s0f0`; leaving it in lets NCCL mix a 400 Gb/s path into the
same collectives.

---

## 4. IMEX / multi-node NVLink — the other expensive one

**Symptom.** Multi-node NCCL refuses to initialise:

```
torch.distributed.DistBackendError: NCCL error ... unhandled system error
Last error:
MNNVL (cliqueSize 64) is available but not working on this system.
Check the IMEX channel configuration (/dev/nvidia-caps-imex-channels).
Set NCCL_MNNVL_ENABLE=0 to ignore this issue.
```

**What this means.** `cliqueSize 64` — all 64 GPUs across the 16 nodes are **one
NVLink fabric**, not 16 IB-connected islands. Using it needs NVIDIA IMEX to export
fabric memory between nodes. Without IMEX, `cuMemCreate` on a fabric handle
returns CUDA 801 and multi-node NCCL dies.

**Do not** take the hint in the error message. `NCCL_MNNVL_ENABLE=0` makes it run
by falling back to InfiniBand — it "works", and quietly gives up the rack's NVLink
fabric. That is not a GB300 baseline.

**Fix** (`70-imex.yml`), three parts, all needed:

1. `nvidia-imex-580` (version must match the driver).
2. `/dev/nvidia-caps-imex-channels/channel0` does not exist by default and this
   build of `nvidia-modprobe` has no option to create it. It comes from a module
   parameter:
   ```
   # /etc/modprobe.d/nvidia-imex.conf
   options nvidia NVreg_CreateImexChannel0=1
   ```
   Needs the driver reloaded (`rmmod nvidia_uvm nvidia_drm nvidia_modeset nvidia`
   then `modprobe nvidia`) or a reboot.
3. `/etc/nvidia-imex/nodes_config.cfg` — one IP per line, all 16 nodes,
   **sorted so the file is byte-identical everywhere**. IMEX disables
   communication between nodes whose node maps differ.

**Verify properly.** The daemon exits 0 when the node list is unusable, so systemd
reports success while IMEX is not actually running. Check the domain, not the unit:

```
nvidia-imex-ctl -q     # -> READY
nvidia-imex-ctl -N     # -> all 16 nodes READY
```

---

## 5. Python stack

**venv on shared storage.** `/mnt/dgxc/venvs/torchtitan`. Home directories are
**not** shared between these nodes — only `/mnt/dgxc` is. Anything a job needs must
live there.

**PyTorch.** `torch==2.14.0` from the `cu130` index (driver is CUDA 13.0), aarch64
cp312.

GB300 reports compute capability **`sm_103`**, and this wheel is built for
`['sm_80','sm_90','sm_100','sm_110','sm_120']` — no `sm_103`. It works anyway;
verified with a real single-GPU bf16 matmul at **1935 TFLOP/s** before trusting it.
Worth re-checking on any torch upgrade.

**FlashAttention.** `ModuleNotFoundError: No module named 'flash_attn'`, raised
from torch's own `torch/nn/attention/_fa4.py`, which wants
`flash_attn.cute.interface` on Blackwell.

The trap: `pip install flash-attn` gives **FA2** (PyPI's latest is 2.8.3.post1,
source-only, no aarch64 wheel, hours to build) and FA2 has no `cute` submodule.
FA4 is a **separate package**, pure Python over CuTe DSL, no long build:

```bash
pip install --pre "flash-attn-4[cu13]"     # --pre required: 4.0.0bNN only
```

**Assets.** Tokenizer + config for the model:

```bash
python scripts/download_hf_assets.py --repo_id openai/gpt-oss-20b --assets tokenizer config
```

C4 is **streamed** from HF at train time (`streaming=True`), not downloaded — so
every compute node needs outbound HTTPS to `huggingface.co` during the run.

---

## 6. Config deviations from stock `gpt_oss_20b`

Only one is a real change to how the model runs.

### Required: `--training.disable_cuda_graphs`

Not a tuning choice. The stock config pairs varlen attention with CUDA graph
capture, but the varlen metadata (`cu_seqlens`) changes length as documents pack
differently each step, so capture fails on step 2:

```
ValueError: CUDA graph tensor inputs must keep the same shape, dtype, and device,
but input 4 changed from (torch.Size([21]), torch.int32, ...) to (torch.Size([26]), ...)
```

The config's own docstring notes CUDA graphs "require fixed-shape inputs".

### Benchmark-only: `--training.steps 50`

Default is 10000. Throughput is unaffected, but the LR schedule is auto-clamped:

```
Warmup steps (2000) exceed total steps (50). Adjusting warmup steps to 50.
Warmup (50) + decay (40) steps exceed total steps (50). Adjusting decay steps to 0.
```

So the whole run is warmup and **the loss curve is not on the real schedule**.
Fine for measuring throughput; do not read the loss descent as meaningful.

### Everything else is registry default

Verified from the run log, not assumed:

```
Building device mesh with parallelism: pp=1, dp_replicate=1, dp_shard=64, cp=1, tp=1, ep=1
Successfully created meshes with active dimensions: ['batch', 'loss', 'fsdp']
Applied FSDP to the model
Applied FullAC activation checkpointing to the model
Optimizer AdamW: lr 0.0008, betas (0.9, 0.95), eps 1e-08, weight_decay 0.1, fused
Trainer is initialized with local batch size 1, global batch size 64,
  gradient accumulation steps 1, sequence length 8192
```

Pure FSDP sharded over all 64 ranks (`data_parallel_shard_degree=-1` resolving to
world size). No TP, PP, CP or EP.

---

## 7. Measurement artifact worth knowing

The first 64-GPU run used `--metrics.log_freq 1` and `OMP_NUM_THREADS=8`. Both
deviate from stock (`log_freq` defaults to 10; torchrun sets `OMP_NUM_THREADS=1`).
Re-running at stock defaults was **faster**:

| run | log_freq | OMP_NUM_THREADS | per-GPU TFLOP/s |
| --- | --- | --- | --- |
| job 38 | 1 | 8 | 277.4 |
| job 39 | 10 (default) | 1 (default) | **287.8** |
| job 40 | 10 (default) | 8 | 287.0 |

A third run isolated it: **`log_freq` was the entire cause** (~3.7%), and
`OMP_NUM_THREADS` 1 vs 8 is noise (287.8 vs 287.0, ~0.3%). `log_freq=1` forces a
per-step synchronisation.

**Use stock `log_freq` when benchmarking.** The launch script now defaults to 10
and no longer forces `OMP_NUM_THREADS`.

The headline baseline is **job 39** — full stock defaults apart from the two
documented deviations in §6.

---

## Reproducing

```bash
sbatch gb300/gpt_oss_20b_64xgb300.slurm
```

Preflight, if something looks wrong:

```bash
nvidia-imex-ctl -q                                   # READY
ls /sys/class/infiniband                             # 4x ibP* + roce*
grep online_movable /proc/cmdline                    # present
grep ConstrainCores /etc/slurm/cgroup.conf           # =no
srun -N1 --gres=gpu:4 python -c "import torch; torch.zeros(1).cuda()"
```
