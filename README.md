# CDRec
## RUN
python -u main.py --model DisenCDR --dataset sport_clothing
nohup python -u main.py --model Base --dataset sport_clothing >run.log 2>&1 &

CUDA_VISIBLE_DEVICES=0 nohup python -u main.py --model CD_CDR --dataset Amazon2014 --domains Clothing_Shoes_and_Jewelry Sports_and_Outdoors >run.log 2>&1 &
CUDA_VISIBLE_DEVICES=0 nohup python -u main.py --model DiffCDR --dataset Amazon2014 --domains Sports_and_Outdoors Clothing_Shoes_and_Jewelry >run1.log 2>&1 &

## Cached Target Patterns

The `target_pattern_pseudo/` directories under `data/` are stored with Git LFS
for Cloth-to-Sport, Phone-to-Elec, and Book-to-Music. They include generated
text, semantic embeddings, completion metadata, transfer checkpoints, and
pseudo interaction rankings. Other data directories and runtime logs are
not included.

Install Git LFS before cloning. In an existing clone, download the cache files
with:

```bash
git lfs install
git lfs pull
```

These files must be downloaded from LFS to use the cached results; Git pointer
files alone are not the generated data. Retain the metadata and completion
files together with the embeddings to preserve their ID alignment.
