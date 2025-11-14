# Lambda torchtitan fork

## Set up

Install torch, use
```
pip install torch --index-url https://download.pytorch.org/whl/cu130
```

```
cd torchtitan
pip install -r requirements.txt
pip install torchao
pip install -e .
```

## Running lambda fork

```
git checkout main
sudo nvidia-smi boost-slider --vboost 1
export PYTORCH_ALLOC_CONF=expandable_segments:True
torchrun --nproc-per-node=gpu -m torchtitan.train <config file>
```

Optimized config files can be found under [./configs](./configs)

In order to run 16xB200 configurations, instead use run_train_c0.sh and run_train_c1.sh in the torchtitan directory.
Run the following command on both nodes AT THE SAME TIME, running run_train_c0.sh on node-001 and run_train_c1.sh on node-002:
```bash
./run_train_c<0 or 1, depending on the node you are on>.sh --config <config file you want to use for multi-node setup>
```

The 16xB200 config files can also be found under [./configs](./configs).

## Running baselines

```
git checkout torchtitan-e7ee95a
sudo nvidia-smi boost-slider --vboost 0
export PYTORCH_ALLOC_CONF=expandable_segments:True
torchrun --nproc-per-node=gpu -m torchtitan.train <config file>
```

Config files for baselines can be found under: [./torchtitan/models/llama3/train_configs](./torchtitan/models/llama3/train_configs)

