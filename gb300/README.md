# GPT-OSS-20B baseline on 64x GB300

Out-of-the-box baseline for `gpt_oss_20b` across 16 nodes x 4 GB300 on the
yqb01-qa01 cluster, 2026-09-06.

```bash
sbatch gb300/gpt_oss_20b_64xgb300.slurm          # 50 steps
STEPS=100 sbatch gb300/gpt_oss_20b_64xgb300.slurm
```

## Result (job 39, 50 steps, stock defaults)

Steady state = steps >=20.

| | |
| --- | --- |
| per-GPU | **287.8 TFLOP/s** |
| per-GPU | ~9,190 tokens/s |
| **64 GPUs** | **18.42 PFLOP/s**, ~588,000 tokens/s |
| MFU | 11.5% |
| memory | 18.46 GiB/GPU (6.68%) |
| loss | 12.71 -> 7.10 |

An earlier run (job 38) reported 277.4 TFLOP/s because it used
`--metrics.log_freq 1` and `OMP_NUM_THREADS=8`. A third run isolated it:
`log_freq=1` costs ~3.7% through per-step synchronisation, and OMP_NUM_THREADS is
noise. Job 39 is the number to quote. See
[DEBUG_FOR_BASELINE_64xGB300.md](DEBUG_FOR_BASELINE_64xGB300.md) §7.

Note the LR schedule is auto-clamped at 50 steps (warmup 2000 -> 50, decay -> 0),
so the whole run is warmup and **the loss curve is not on the real schedule**.
Fine for throughput; do not read the loss descent as meaningful.

`tflops` in the torchtitan log is **per GPU**: 278.2 / 0.1112 MFU = 2502 TFLOP/s,
which is GB300 dense bf16 peak. Multiply by 64 for the cluster figure.

Single node (4 GPUs) on the same config reaches 333.4 TFLOP/s/GPU (13.3% MFU), so
scaling 4 -> 64 GPUs holds **86.3%** of per-GPU throughput.

Memory at 6.68% and MFU at 11% both say the same thing: this config is small for
the hardware. `local_batch_size=1` at `seq_len=8192` is the stock setting and was
left alone on purpose - it is the baseline, not a tuned run. Raising batch size is
the obvious first lever.

## The one deviation from stock

`--training.disable_cuda_graphs` is **required**, not a tuning choice. The stock
config pairs varlen attention with CUDA graph capture, but the varlen metadata
(`cu_seqlens`) changes length as documents pack differently each step, so capture
fails on step 2:

```
ValueError: CUDA graph tensor inputs must keep the same shape, dtype, and device,
but input 4 changed from (torch.Size([21]), torch.int32, ...) to (torch.Size([26]), ...)
```

Everything else is registry default: `local_batch_size=1`, `seq_len=8192`, FSDP
over all 64 ranks (`data_parallel_shard_degree=-1`), `expert_parallel_degree=1`,
`FullAC`, C4 streamed from HF, AdamW lr 8e-4.

## Environment

- `/mnt/dgxc/venvs/torchtitan` - shared across all nodes (home dirs are **not** shared)
- torch 2.14.0+cu130, aarch64, driver 580.173.02
- `flash-attn-4[cu13]` (pre-release). Torch's FA4 registry wants
  `flash_attn.cute.interface` on Blackwell; PyPI's `flash-attn` is FA2 and
  source-only, but FA4 ships as a separate pure-Python package - no long build.
- GB300 is `sm_103`; this torch is built for `sm_100`/`sm_110`. Binaries run fine
  (verified 1935 TFLOP/s single-GPU bf16 matmul).

## Cluster prerequisites this run depended on

All bare metal, none present initially. Ansible for these lives in
`~/cluster-setup` on 00057.

**IMEX / multi-node NVLink.** These 16 nodes are one 64-GPU NVLink fabric, not 16
IB-connected islands. NCCL refuses to start without it:

```
MNNVL (cliqueSize 64) is available but not working on this system.
Check the IMEX channel configuration (/dev/nvidia-caps-imex-channels).
```

Needs `nvidia-imex-580`, the `NVreg_CreateImexChannel0=1` module parameter (the
channel device is not created by default and this `nvidia-modprobe` has no option
for it), and a byte-identical sorted node list on every node. `NCCL_MNNVL_ENABLE=0`
makes the error disappear by falling back to InfiniBand - it "works" and quietly
gives up the rack fabric, so do not use it for a baseline.

Note the NVL72 switch trays are outside the compute node, so `lspci` shows no
NVSwitch and `nvidia-fabricmanager` correctly reports "Nothing to do". That is
unrelated to IMEX, which is what makes the rack-level fabric usable.

**InfiniBand.** 4x ConnectX-8 at 800 Gb/s per node, but `mlx5_ib` lives in
`linux-modules-extra`, which was not installed - so there was no
`/sys/class/infiniband` at all and NCCL would have fallen back to TCP over the
400 GbE. `NCCL_IB_HCA` in the launch script names the four IB devices and
deliberately excludes `roceP22p1s0f0`, the Ethernet port.

**Slurm cgroup.** `ConstrainCores` must be off. Slurm derives `cpuset.mems` from
the CPU NUMA nodes (`[0-1]`), but these machines expose
`0-1,3-9,11-17,19-25,27-33` - the extras being coherent GPU memory the driver
allocates from during init. Confined to `[0-1]`, CUDA reports "No CUDA GPUs are
available" inside Slurm while working fine over SSH.
