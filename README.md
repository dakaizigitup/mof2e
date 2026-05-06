# mof2e — Modified fairchem-core for MOF / CO2–H2O cooperative adsorption

This repository contains:

1. A **modified fork of `fairchem-core 1.10.0`** (under `src/fairchem/core/`)
   with custom models, trainers, and dataset classes for MOF adsorption
   energy prediction under cooperative gas conditions.
2. The **complete experiment workspace** with both legacy backbone training
   (eSCN / EquiformerV2 / GemNet / SchNet / PaiNN / SphereNet / MultiStream)
   and the **V9 causal-embedded model** (`causal_model/` + `causal_main.py`).

> Upstream fairchem-core documentation: see [`README_FAIRCHEM_UPSTREAM.md`](README_FAIRCHEM_UPSTREAM.md).

---

## Repository layout

```
mof2e/
├── src/fairchem/core/         Modified fairchem-core (the *active* library)
│
├── main.py                    Legacy backbone training entry
├── causal_main.py             V9 causal training entry
├── causal_model/              V9 model + trainer + configs + eval + data tools
├── mof2e/                     Per-backbone experiment YML
│   ├── eSCN/
│   ├── EquiformerV2/
│   ├── GemNet/
│   ├── SchNet/
│   ├── PaiNN/
│   ├── SphereNet/
│   ├── MultiStream/
│   └── ...
├── configs/                   Upstream OCP baseline configs
├── lora_model/                LoRA training code (weights .pt excluded — see below)
├── irm_trainer.py             Early IRM trainer
├── transform_to_lmdb.py       Data preprocessing
├── extract_data.py / extract_info.py / debug_dataset.py / convert_to_ase.py
├── tests/                     Tests
├── docs/                      Upstream + project docs
├── results/                   Small experiment results
├── phast-main/                PhAST third-party reference
│
├── env.gpu.yml                Conda env recipe
├── requirements.txt           Pinned pip deps
├── pyproject.toml             Editable install descriptor
└── .gitignore
```

---

## Deployment to a fresh server

### Path A — Same-arch machine (e.g. another AutoDL node), with pre-built env tarball

Fastest, ~5 min total. Use this when you have `fairchem_env.tar.gz` ready
to upload (created by `conda-pack -n fairchem -o fairchem_env.tar.gz`).

```bash
# 1. Upload fairchem_env.tar.gz to ~/ on the new server (xftp/scp)

# 2. Restore the conda env from the tarball
mkdir -p ~/anaconda3/envs/fair-chem
tar -xzf ~/fairchem_env.tar.gz -C ~/anaconda3/envs/fair-chem
source ~/anaconda3/envs/fair-chem/bin/activate
conda-unpack       # required after conda-pack to fix absolute paths

# 3. Generate SSH key on this server and add to GitHub
ssh-keygen -t ed25519 -C "you@example.com" -f ~/.ssh/id_ed25519 -N ""
cat ~/.ssh/id_ed25519.pub        # paste at https://github.com/settings/keys

# 4. Clone and install the modified fairchem on top
git clone git@github.com:dakaizigitup/mof2e.git
cd mof2e
pip install -e .                 # editable install — git pull updates fairchem live
```

### Path B — Fresh build from recipe (any machine with matching CUDA driver)

Slower (~15-20 min), but no tarball needed.

```bash
git clone git@github.com:dakaizigitup/mof2e.git
cd mof2e
conda env create -f env.gpu.yml
conda activate fair-chem
pip install -e .
```

---

## Running experiments

### Legacy backbone training (e.g. eSCN baseline)

```bash
python main.py \
  --config-yml mof2e/eSCN/eSCN_stage1.yml \
  --mode train
```

### V9 causal-embedded model

```bash
python causal_main.py \
  --config-yml causal_model/config_causal/v9/eSCN_v9_s1.yml \
  --mode train
```

V9 design docs: see [`creatorMd/`](creatorMd/) once added (Proposal9 / Plan_V9).

---

## What is NOT in this repo

| Asset | Where | Reason |
|---|---|---|
| Training data (`data/`) | rsync from server A or object storage | hundreds of GB |
| Checkpoints (`checkpoints/`) | rsync from server A | tens to hundreds of GB |
| Pre-built conda env tarball (`fairchem_env.tar.gz`) | xftp directly between servers | 13 GB, exceeds GitHub 100 MB single-file limit |
| `lora_model/ckpt_is2re/Equiformer_V2_Direct.pt` | xftp / fairchem-core release | 108 MB single file |
| Logs / runs / wandb | runtime artifacts | regenerated each run |
| `__pycache__/`, `*.pyc`, build artifacts | `.gitignore` | generated |

After git clone, manually transfer the above as needed:

```bash
# On server B (after git clone)
mkdir -p data checkpoints lora_model/ckpt_is2re

# Get data + checkpoints from server A
rsync -avz dell@server-A:/home/dell/autodl-tmp/lorafair/data/    ./data/
rsync -avz dell@server-A:/home/dell/autodl-tmp/lorafair/checkpoints/  ./checkpoints/

# Get LoRA weight if needed
scp dell@server-A:.../Equiformer_V2_Direct.pt  ./lora_model/ckpt_is2re/
```

---

## Iterative development workflow

After first deployment, only git is needed for code updates:

```bash
# Dev server (where you edit)
git add ... && git commit -m "..." && git push

# Any other server
cd ~/mof2e && git pull
# Changes take effect immediately — pip install -e . links to source
```

You **never need to re-build the env or re-upload the tarball** unless you
add new conda/pip dependencies. In those rare cases, run
`conda env update -f env.gpu.yml` after pulling.

---

## Versioning notes

- Base library: `fairchem-core 1.10.0` (upstream from PyPI)
- Local additions/modifications: see `git log` on `src/fairchem/core/`
- V9 design notes: `creatorMd/9MOF_Causal_Embedded_Model_Proposal9.md`,
  `creatorMd/10MOF_Causal_Embedded_Model_Implementation_Plan_V9.md`
