# LambdaLabsML/torchtitan

## Virtual Environment set up

Create your venv:
```
python -m venv .venv
source .venv/bin/activate
```

Install torch:
```
pip install torch --index-url https://download.pytorch.org/whl/cu128
```

Install dependencies & torchtitan:
```
cd torchtitan
pip install -r requirements.txt
pip install torchao
pip install -e .
```

## Running lambda fork

```
sudo nvidia-smi boost-slider --vboost 1
export PYTORCH_ALLOC_CONF=expandable_segments:True
torchrun --nproc-per-node=gpu -m torchtitan.train <config file>
```

Optimized config files can be found under [./configs](./configs)

The config files under this directory are generally named like:

```
<model type>_<size>-<num gpus>x<gpu type>-<seq len>.toml
```

Here are a couple examples:
- `./configs/llama3_405b-32xgb300-8k.toml`: the Llama 405b recipe for 32 GB300 training on 8k sequence length.
- `./configs/llama3_70b-16xb200-64k.toml`: Llama 70b recipe for 16 B200 gpus with 64k sequence length

## Submitting to a slurm cluster

We provide our `train.sbatch` to launch jobs on slurm clusters, here's a command to launch the training job:
```
sbatch --nodes <num nodes> train.sbatch --job.config-file <config file>
```
