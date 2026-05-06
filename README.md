# mof2e — Modified fairchem-core for MOF / CO2–H2O cooperative adsorption

This repository contains a **modified fork of `fairchem-core 1.10.0`** with custom
models, trainers, and dataset classes added for MOF adsorption energy prediction
under cooperative gas conditions.

> Project code (V9 causal-embedded model, experiment configs, docs) will be added
> in a later commit. This first commit ships only the modified library + env recipe.

---

## Repository layout

```
mof2e/
├── env.gpu.yml          # Conda env recipe (PyTorch 2.4.0 + CUDA 12.1 + e3nn + PyG)
├── requirements.txt     # Pinned pip deps
├── pyproject.toml       # Makes the modified fairchem installable via `pip install -e .`
├── src/
│   └── fairchem/
│       ├── __init__.py
│       └── core/        # ← The actually modified library (used at runtime)
│           ├── models/        (incl. multi_stream_*, schnet_global,
│           │                   equiformer_v2/atomic_emb*, .../global_emb*,
│           │                   .../equiformer_lora, escn/escn_lora, etc.)
│           ├── causal_analysis/
│           ├── trainers/
│           ├── datasets/
│           ├── common/
│           └── ...
└── README.md
```

---

## Deployment to a fresh server

There are two paths depending on what you have available.

### Path A — Same-arch machine (e.g. another AutoDL node), with pre-built env tarball

Fastest, ~5 min total. Use this when you have the `fairchem_env.tar.gz`
(`conda-pack`'d archive of the working env) ready to upload.

```bash
# 1. Upload fairchem_env.tar.gz to ~/ on the new server (xftp/scp/rsync).

# 2. Restore the conda env from the tarball
mkdir -p ~/anaconda3/envs/fair-chem
tar -xzf ~/fairchem_env.tar.gz -C ~/anaconda3/envs/fair-chem
source ~/anaconda3/envs/fair-chem/bin/activate
conda-unpack       # required after conda-pack to fix absolute paths

# 3. Clone this repo and install the modified fairchem on top
git clone https://github.com/dakaizigitup/mof2e.git
cd mof2e
pip install -e .   # editable install — git pull will update fairchem live
```

### Path B — Fresh build from recipe (any machine with matching CUDA driver)

Slower (~15-20 min), but no tarball needed.

```bash
git clone https://github.com/dakaizigitup/mof2e.git
cd mof2e

conda env create -f env.gpu.yml
conda activate fair-chem

pip install -e .
```

---

## Iterative development workflow

After the first deployment, you only need git for code updates:

```bash
# On dev server (where you edit code)
git add src/fairchem/core/...
git commit -m "..."
git push

# On any other server with the env already set up
cd ~/mof2e && git pull
# Changes take effect immediately — pip install -e . links to source.
```

You **never need to re-build the env or re-upload the tarball** unless you add
new conda/pip dependencies to `env.gpu.yml` or `requirements.txt`. In those
rare cases, run `conda env update -f env.gpu.yml` after pulling.

---

## What is NOT in this repo (and why)

- `data/` and `checkpoints/` — too large (hundreds of GB); transferred via
  rsync / object storage, not git
- `fairchem_env.tar.gz` — 7.1 GB; transferred via xftp directly between
  servers, not stored on GitHub (exceeds 100 MB single-file limit)
- `__pycache__/`, `*.pyc`, build artifacts — see `.gitignore`

---

## Versioning notes

- Base library: `fairchem-core 1.10.0` (upstream from PyPI)
- Local additions/modifications: see `git log` on `src/fairchem/core/`
