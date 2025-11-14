# Lambda torchtitan fork

## Running lambda fork

```
git checkout main
sudo nvidia-smi boost-slider --vboost 1
export PYTORCH_ALLOC_CONF=expandable_segments:True
torchrun --nproc-per-node=gpu -m torchtitan.train <config file>
```

Optimized config files can be found under [./configs](./configs)

## Running baselines

```
git checkout torchtitan-e7ee95a
sudo nvidia-smi boost-slider --vboost 0
export PYTORCH_ALLOC_CONF=expandable_segments:True
torchrun --nproc-per-node=gpu -m torchtitan.train <config file>
```

Config files for baselines can be found under: [./torchtitan/models/llama3/train_configs](./torchtitan/models/llama3/train_configs)

