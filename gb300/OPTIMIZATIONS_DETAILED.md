# DeepSeek-V4 Flash on GB300: every optimization in depth

The whitepaper summarizes the most significant changes behind the DeepSeek-V4
Flash results on GB300. This document goes through every optimization we tried,
one section per pull request. Sections are named after the branch the change
was reviewed on (`split/NN-<name>`, merged into `main` as PRs #25 through #70);
the number is the order the changes were built in, and each branch holds exactly
one change. The commands that reproduce the baseline and the best run are in the
README on the `lambda-moe-tflops` branch, under "Recreating our runs".

## split/01-flex-max-autotune-per-layer

optimization to shorten start up time. More of a fix to a previous patch we had to let attention run with DSv4 Flash on the GB300s. Autotuning was happening with our pinned settings without finding meaningful increase in throughput or training performance, and taking a ton of setup time. By enabling this autotune per layer setting so that we can disable it, it allowed a several minute reduction (228 seconds) in startup time per distinct tensor shape, basically bypassing all this unnecessary searching. Throughput between tuning set and not set is almost the same (92.01 vs 92.15 tflops/sec/gpu), while the setup time is reduced a ton. Allows us to iterate on experiments faster with trivial change in actual training performance.

## split/02-restore-minimal-async-ep

brought back the code that enables MinimalAsyncEP as a backend. Since it was deprecated by the Torchtitan codebase due to lack of maintenance, read the implementation from Megatron and implemented it again. When using EP, different experts are hosted on different GPUs. Normal routing of tokens works by doing 1) making routing decision to certain experts and calculate their routing values, 2) send the tokens to the appropriate experts on their GPUs, 3) run the tokens through the experts’ networks, and 4) return the vectors back to the original routing GPU and form the weighted sum. Since the CPU needs to wait until all the GPUs return their results, and weighted routing inherently introduces load imbalance, experts/GPUs with smaller batches are left waiting on all the experts/GPUs with larger batches to process the tokens’ hidden vectors and send it back in one all-to-all once everybody is done. The synchronization with the CPU for it to make correct decisions on token dispatch, then, becomes a significant bottleneck.

Our implementation instead allows for the GPUs to preallocate memory to communicate token counts for each other rather than relying on the CPU. Buffers in memory are created in order to estimate how many tokens will be dispatched to peer GPUs, and extra GPU kernels are scheduled to send and receive these tokens once the GPUs are done processing them. The preemptive nature of estimating token count and dispatch size on the GPU avoids needing synchronization with the CPU to make token dispatch decisions due to dynamic loads.

## split/03-gb300-launcher

slurm file that adds specific NCCL/IB env flags for 64xGB300 clarification.

## split/04-gb300-bf16-recipe

Unpinning tile size to let the autotuner figure out optimal size showed that all sizes should be set to 16. Was all in an effort to fix a CUDA error. This was undone in later PRs and just left to autotuner.

## split/05-selective-ac-sweep-config

Selective AC sweep to try and make it work for DSv4 Flash. Did not work, showing how we could only work with Full AC. Maybe a tiny bit of tuned AC could make performance better, but was not worth the implementation to try and save 5% of tensors for the backward pass.

## split/06-indexer-deterministic-select

Fixes a bug that was happening with Selective AC and batch sizes larger than 1. AC throws away tensors from the forward pass that aren’t selected for computation in the backward pass. Since choosing top-k indices for tokens was previously not stable, indices with the same score (more likely in lower precision, since similar values can be rounded to the same value in FP8/BF16 compared to FP32/FP64) can be chosen by the forward and backward pass if they are in the top-k. For example, in a top_k=8 scenario if the 8th and 9th tensors both have the same score, we could originally choose the 8th tensor in the forward pass and then choose the 9th tensor in the backward pass. However, since we lost the original math from the 9th tensor because we threw it away with AC, we run into an error where it’s no longer in memory. The stable selection we added automatically chooses the tensor with the smallest index, making it deterministic but consistent. Removes this problem of missing tensor in the backward pass with AC. Originally, I thought this was some weird collusion between batch size > 1 and AC since it was the only time I observed the error. However, this could be due to other semantics where processing more batches of data at once makes this selection error more likely to happen due to averaging and more similar scores in the top_k.

## split/07-cudnn-dsa-backward

completely rewrote the backward pass in cuDNN for Deepseek Sparse Attention (DSA).

## split/08-cudnn-dsa-forward

rewrote the forward pass in cuDNN for Deepseek Sparse Attention (DSA). Was smaller since it just reuses the backward pass logic. TODO

## split/09-fp32-matmul-precision-knob

Precision knob that was eventually deprecated. Allowed to change precision between TF32 and bf16x9 for CUDA matmuls. I guess BF16x9 (default for matmuls) is overkill when we could use FP32/TF32. It was in a very niche function in utils.py and didn’t do anything.

## split/10-fused-output-rope

Was technically two optimizations in how DeepSeek v4 models were 1) handling sliding window attention and 2) fusing the outputs of RoPE. Taken from Megatron issue #7036.

For the RoPE fix, the last 64 numbers of attention head dimension of 512 is used for positional encoding that RoPE uses. The original Torchtitan code was reading all 512 numbers, creating a whole new tensor where the only thing that changes was the inversing of the last 64 dimensions (needed for RoPE in DSv4), and then the output projection. Not only is this wasting memory by creating a whole new tensor (the code kept the original tensor, two copies of dim=512), but it reads and writes all 512 dimensions. Our fix inverses the last 64 used for positional encoding IN PLACE. Now, only reads and writes 64 numbers, as well as not creating an extra tensor in memory. Since the backward pass still needs the original positional encodings, we flip it back in place to the original vector during the backward pass.

For the sliding window attention fix, get_window_topk_idxs was used to compute which key positions SWA can look at. However, it builds it from scratch every time it is called, which is frequently during SWA (86 times with our sequence length of 8k). Since it depends on sequence length and window size, which are static, we only need to call this function once, store the possible indices, and then use the expand() function to view them later on, which is a free viewing of the indices without need for recomputation.

## split/11-fsdp-dense-never

Allows us to ignore the reshard for dense layers only, while resharding just the expert layers of our MoE models. When resharding after the forward pass is enabled, we need extra all-gathers to grab the layers from every other GPU to appropriately update our weights during the backward pass. There is more communication overhead, but we save a ton of memory since we only hold the ~1 layer in memory during any point of the forward pass. With reshard disabled, our weights accumulate since we keep the layers on the GPU moving through the forward pass. This means we have less memory we can use, but we reduce communication overhead significantly.

Since DSv4 Flash is a large MoE model, we cannot get away with disabling the reshard after the forward pass. The model is simply too large to keep all the layers in memory during the forward pass. However, by resharding just the expert layers after the forward pass, we can disable resharding for the dense layers, allowing for some optimization. We have enough memory on the GB300s to keep the dense layers and remove the gathers needed for them.

## split/12-fp8-expert-allgather

Taken from Megatron issue #5470. Basically, we gather and communicate expert layers across GPUs in FP8, even though the master weights are kept in higher precision for accuracy. We lose some accuracy on weight updates since the conversion from higher precision → FP8 → back to higher precision introduces error due to rounding, but a negligible amount that trivially affects loss. The lower precision all-gather was absent from torchtitan and poorly implemented in torchao with sm_103, so was taken from Megatron instead.

## split/13-maep-4-receive-slots

Receive-slot count increased from 2 to 4 using minimal Async EP. Under FullAC the experts' saved input aliases the slot a dispatch returned and the wgrad GEMM reads it in backward; every comm op (fwd dispatch, fwd combine, bwd combine, bwd dispatch) rotates the pool, so with 2 slots the backward ops rewrite it first (a 0.12% gradient error, jobs 754 vs 768). 4 slots keep the alias intact through a block's backward.

## split/14-cudnn-indexer

passes DSA indexer to cuDNN instead of using suboptimal Torchtitan implementation. Original implementation takes the tokens from the query and einsum’s them with every KV vector. Fills out a table in memory with all the results, sorts all of them and chooses the top k values. Passing the work to cuDNN combines the scoring and selection process, and then avoids writing every single calculation to a table in memory.

## split/15-fp8-dense-torchao-custom

Tried getting FP8 to work with dense layers with TorchAO and sm_103. Took inspiration from Megatron to try and make it work, but didn’t end up working well.

## split/16-te-dense-linear

Since the previous attempt at converting dense layers to fp8 wasn’t working, took implementation directly from TE and installed it in the venv we were using. Created file at torchtitan/quantization/te_linear.py in our repo to do it. Saw great performance boost.

## split/17-grouped-lowrank-proj-bmm

torch.einsum implementation is naturally tedious in copying large tensors to rearrange their memory layout before and after matrix multiplication. In our implementation, we create an empty matrix buffer according to the expected output dimensions, and write out our batched mm results in the correct order. Avoids unnecessary memory read/copies in the forward and backward passes.

## split/18-indexer-inplace-rope

Since CSA compressing only 4 tokens still leaves "too many to attend to" in DSA, they introduce Indexer. This chooses which summaries of tokens each query reads during CSA. The indexer here also has a query RoPE, and a separate RoPE function for figuring out which token summaries the indexer should return. Had the same suboptimal inversing that was fixed in split/10, so was fixed here too.

## split/19-microbatch-variant-configs

Batch size sweep after saving a lot of memory via TE quantization implementations that were actually working. Found a batch size of 7x started working (compared to 6x) and saw a small performance boost.

## split/20-deepep-variant

Trying to make DeepEP work again. Was trying to offload MinimalAsyncEP since its deprecated upstream. However, DeepEP still underperformed. This PR is just to note that it was tried a few times and still didn't work.

## split/21-cublas-13.8-preload

preloads TE dependencies on PATH in our slurm launcher.

## split/22-te-grouped-experts

Implementing TE’s MXFP8 GroupedExperts function, since again the torchtitan/torchao support wasn’t actually helping increase tflops. Made sure MMs are actually being quantized to FP8 properly and being used in optimized sm_103 architecture. Also, torchtitan had several repeated quantizations that were removed.

## split/23-te-fused-mhc

Fuses mHC function according to TE’s implementation. First, before attention, TE does RMSNorm and projections as one, reducing copying new tensors into memory for each. After attention is calculated, TE calculates the expanded output and mixed residual streams and then writes the addition of the two directly into an output tensor, rather than creating two tensors in memory for each calculation and then a final new tensor for the addition of the two. Less moving and storing intermediate calculations in memory.

## split/24-optimizer-state-offload

Taken from Megatron issue #6244. Saved 33GB per GB300, a massive saving that allowed for increased batch size, which is a primary path for improving training efficiency. AdamW stores two values on top of the weights and their corresponding gradients, in the same precision: 1) a moving avg of past gradients, keeping track of the direction weight updates have been taking, and 2) a moving avg of squared gradients, which helps scale future updates. Although useful for optimized weight updates, it takes a ton of extra overhead memory. Our change offloads these history states to CPU memory, and transferring them to the GPU only when we are updating our weights. This opens up tons of free memory during the forward and backward pass. To cover up the costs associated with all the memory transfers between GPU → CPU and CPU → GPU, we overlap CUDA threads doing GPU work and HtoD/DtoH transfers, as well as behind the forward pass.

## split/25-block-input-offload

Similar fix from AdamW. AC saves memory by throwing away intermediate activations, but we still need the block inputs for recalculation during the backward pass. Since they aren’t needed immediately, we offload block inputs needed for activation calculation to CPU memory, and bring them back when we are recomputing in the backward pass. Saves a good amount more memory.

## split/26-sac-ep1-experiment-configs

Reattempting Selective AC with all our saved memory. Again was not able to get it working.

## split/27-te-reduce-amax

Basically, GPUs sync up on their local maxima in a tensor for appropriate scaling to FP8 values. We took influence from TE/Megatron and removed this syncing. Thus, each GPU designates a local scaling rule based on its local maximum value in any given tensor, rather than the absolute maximum across all the tensors in any expert/GPU. Removes this communication cost DtoD.

## split/29-c4-local-dataset

Local copy of data instead of streaming it in from HF. We have done this before.

## split/30-hybridep-variant

Same attempt as split/20-deepep-variant, except with HybridEP. Saw reduction in tflops by 2-3%. Bummer that we couldn’t get one of these backends working properly, especially since HybridEP was supposed to work well on GB300/sm_103

## split/31-bounded-swiglu

Investigation revealed that the expert's input would iterate over the entire buffer capacity, even though only ~50% of the buffer was real tokens that grouped GEMMs stayed bounded to. Thus there was a bunch of computation being made from activations on input the grouped GEMMs weren't even computing. Thus, we bounded SwiGLU based on what grouped GEMMs actually does, saving unnecessary compute from happening. Saw a solid improvement in tflops (3%) and slight reduction in memory utilization.

## split/32-fsdp-prefetch-depth

allows us to configure how many layers we want to prefetch during FSDP. Original prefetch implementation allowed fetching of the next layer, overlapping forward/backward pass of current layer with all-gather of getting the next layer. Found optimal balance with two layers, where we can fetch an extra layer while still having enough memory to keep AC, batch size and other parameters the same.

## split/33-balanced-routing-configs

Implements naive load balancing. Simply routes tokens uniformly across the experts, ignoring router decisions. Although it allows for enormous boost in performance, the router isn’t actually influencing the decisions. Not real training, and must be noted. If implemented in a valid manner, where more targeted decisions are made towards routing the preserve decision making and acknowledges sacrifices in accuracy, then it is a viable option. However, torchtitan’s naive moe.router._debug_force_load_balance does not support actual MoE training. Gate scores never actually learn which experts to choose.

## split/34-maep-copy-geometry

Exposes env knobs that allow CUDA threads to optimally copy rows of tokens in MinimalAsyncEP per GPU. Was originally static; this dynamic nature allows for a small increase in tflops from 578 to 588 by more optimally copying token vectors between GPUs to their memory buffers in minimal async EP.

## split/35-dual-microbatch

splits batches into two. Idea was while a GPU is handling the first half of the data, the other half gets sent elsewhere to be processed by more idle GPUs and then brought back. Degraded perf heavily, but thought it was an interesting idea from Claude.

## split/36-maep-pool-factor

Change implemented on the assumption of balanced loads, making this optimization sketchy. Minimal Async EP previously reserved a memory buffer on peer GPUs assuming the worst case scenario, which is every EP rank sends their tokens to a single EP rank. Saw a reduction in memory utilization when enabled.

## split/37-ep-degree-configs

grid-searched EP values up to world size in order to see if more EP was better. In MLPerf, had seen scenarios where setting EP == world_size while using HybridEP provided the most optimal training performance with NVIDIA’s code. Was not able to find the same performance improvements with torchtitan. Nonetheless, furthers the question as to how HybridEP could help us if properly implemented on torchtitan + GB300s.

## split/38-packed-expert-weights

To supplement FSDP more appropriately. Allows for the 3 learned weight matrices of experts to be packed into one parameter. Allows for direct copy into GPU memory where they will be used instead of into the buffer to ensure correct ordering of weights (see split/39). With this change and the optimized FSDP direct gather in split/39, was able to get batch size = 8 and a great tflops increase.

## split/39-fsdp-direct-gather

Originally, FSDP was gathering the layer shards from all the GPUs and placing it into a buffer on the GPU doing the calculation. It would then copy the shards into the appropriate order in a full-storage allocation to then do the full computation. The new implementation copies each shard directly into the full-storage allocation, instead of putting it in the buffer first. The incoming rows need to form the expected shape, however, so it only works in these eligible cases. However, when it is used, it removes the need to write to the temporary buffer, reading from the temporary buffer and then writing from the temporary buffer to the actual allocation. Improved tflop performance by 6%

## split/40-profiler-with-stack

Added profiling to look at kernel times with the updated stack.

## split/41-dsa-deterministic-knob

Allows for non-deterministic rounding of numbers in precision during gradient calculation. Instead of requiring all threads to finish and sorting the final result to match exact gradient each time, simply adds result of each thread to shared memory allocation as they come in. Saves memory (no individual memory buffer for each thread) and removes gather of all these memory buffers needed to calculate final deterministic result. With this toggled on,

## split/42-fsdp-direct-reduce-scatter

very similar change to split/39 with fsdp direct all gather. There was a temporary memory buffer for packing the gradients to send back to each GPU. Instead, if the gradients are lined up in the correct way, we read the gradient from the result matrix and transfer the gradient directly to the destination device. Skips unnecessary memory copy and allocation of memory buffer across the GPUs.

## split/43-maep-tma-copy

this was super cool, found by Claude! NVIDIA has created the Tensor Memory Accelerator (TMA), a “dedicated hardware engine introduced in NVIDIA Hopper and Blackwell architectures that manages asynchronous, bulk data transfers between global GPU memory and shared memory” (Gemini summary). Using its specialized commands for memory transfer for DtoD (d0.store() and d1.store(), as compared to tl.store() normally), we use the TMA path to transfer the tensors to the same addresses in memory faster. Kind of a crazy find

## split/44-linear-bf16-fp32out

switches the GEMMs to BF16 and accumulation in FP32. Bypasses casting of inputs and original weights to FP32. No performance boost and no reduction in memory, so didn’t really work.

## split/45-compressor-bf16-gemm

Tries the same thing as split/44, but does it in CSA for deepseek v4. When converting groups of 4 tokens into 1 for KV for CSA, we first run a compression matrix multiplication to create the KV vector, and then another matrix multiplication on the available KV vectors to score which ones the query should attend to. Originally we casted the activations and weights up to FP32 and then ran the multiplications. Now, we run the matrix multiplications in BF16 (where the gradient and activations already are) and the accumulate in FP32, and return in FP32. Saw about a 1.5% increase in tflops.

## split/46-mhc-grad-chain

Optimizes how mHC adds together gradients during the backward pass. The same input is used by mHC’s projection, aggregation, and residual path, so each produces a gradient contribution that we need to sum together. Originally, these were combined using two extra passes over the entire input-sized tensor. This means reading and writing a large amount of GPU memory just to add together gradients we have already calculated.

Instead, we use the residual gradient as the starting buffer and let the patched TE kernels add the projection and aggregation gradients directly in-place as they are computed. This removes the two separate addition passes, reducing memory traffic and kernel launches. We do a small amount of extra mHC recomputation during backward to make this work without keeping additional tensors alive.
