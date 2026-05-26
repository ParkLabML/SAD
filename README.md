# Safe Denoiser for Discrete Diffusion Language Models

<p align="center">
  <a href="https://arxiv.org/abs/2605.08116">[Paper]</a> &nbsp;|&nbsp;
  <a href="#citation">[Citation]</a> &nbsp;|&nbsp;
  <a href="#quick-start">[Quick Start]</a>
</p>

> **Safe Denoiser for Discrete Diffusion Language Models**
> [Amman Yusuf](ammanyusuf.com), [Zhejun Jiang], [Mijung Park](https://www.cs.ubc.ca/~mijungp/)
> [ICML 2026]

---

## Overview

We introduce **SAD (Safe Annealed Denoising)**, a training-free safety mechanism for Masked Discrete Language Models (MDLMs) such as [LLaDA](https://github.com/ML-GSAI/LLaDA), [Dream](https://github.com/Dream-org/Dream), and [MDLM](https://github.com/kuleshov-group/mdlm). SAD steers token predictions **away from unsafe content during sampling** — no retraining, no fine-tuning, no classifier in the loop.

The core idea: at each denoising step within a guidance window, we compute the expected denoised embedding under the base model and a pre-built *unsafe negation set*, then push the output embedding away from the unsafe direction by strength β*:

$$E_\text{safe} = E_D + \beta^* (E_D - \hat{E}_{D,\text{unsafe}})$$

<p align="center">
  <img src="assets/safe_denoiser_overview.svg" width="800" alt="SAD overview: the safe denoiser redirects the diffusion trajectory away from unsafe content during a guidance window of timesteps."/>
</p>

*SAD applies during a window of diffusion timesteps (t ∈ C). The base sampler produces toxic outputs; SAD steers towards high-likelihood safe continuations without any model modification.*

---

## Case Studies

SAD blocks jailbreak attacks that fool the baseline defense (DiffuGuard), while the unguarded model complies.

<p align="center">
  <img src="assets/safe_denoiser_case_study.svg" width="800" alt="Two case studies showing SAD refusing jailbreak prompts while DiffuGuard and the vanilla model comply."/>
</p>

**Case Study 01 — WildJailbreakBench attack**: SAD refuses. DiffuGuard partially complies. Vanilla fully complies.

**Case Study 02 — DIJA structured mask attack**: SAD deflects into an innocuous story. DiffuGuard produces borderline content. Vanilla produces explicit content.

---

## Method

The safe denoiser operates as a logit-level hook during the reverse diffusion process:

1. **Unsafe reference set** — a pre-tokenized tensor of unsafe completions, built once per tokenizer from BeaverTails / RealToxicityPrompts / ToxiGen.
2. **Guidance window** — SAD is active only within timesteps `[t_start, t_end]`, leaving early and late denoising unaffected.
3. **Repellency** — at each active step, token log-probabilities are shifted to reduce the likelihood of tokens that appear in the unsafe reference distribution.
4. **Semantic gating** (optional) — a lightweight gate suppresses guidance when the prompt is already on a safe trajectory, preserving generation quality on benign inputs.

The method is model-agnostic: the same hook interface works for LLaDA, Dream, and MDLM. See [`src/third_party/mdlm/repellency/README.md`](src/third_party/mdlm/repellency/README.md) for the full derivation and configuration reference.

---

## Quick Start

The full pipeline runs on a single GPU. Steps 1–4 are all you need to generate safe completions and score them.

### 1. Install

```bash
git clone <this-repo> && cd safe-text-diffusion
python -m venv .venv && source .venv/bin/activate
pip install -r src/requirements.txt

export REPO_ROOT=$(pwd)
export PYTHONPATH=$REPO_ROOT:$REPO_ROOT/src:$REPO_ROOT/src/third_party/mdlm:$PYTHONPATH
```

### 2. Download models and datasets

```bash
python scripts/setup_assets.py --models-dir ./models --data-dir ./data
```

Downloads LLaDA-8B-Instruct, GPT-2-Large (perplexity scoring), and BeaverTails / RealToxicityPrompts / ToxiGen. Run the printed `export` lines before proceeding:

```bash
export CHECKPOINT_PATH=./models/LLaDA-8B-Instruct
export TOKENIZER_PATH=./models/LLaDA-8B-Instruct
export UNSAFE_ARTIFACT_ROOT=./artifacts/unsafe
```

### 3. Build unsafe reference tensors

Built once per tokenizer (< 5 min on CPU):

```bash
python -m unsafe_prep.build_unsafe_artifacts \
  --config configs/unsafe_prep/unsafe_prep.yaml \
  --out $UNSAFE_ARTIFACT_ROOT \
  --force
```

### 4. Generate with SAD

```bash
python -m tools.generate \
  model.family=llada \
  model.checkpoint=$CHECKPOINT_PATH \
  model.tokenizer_name=$TOKENIZER_PATH \
  gen.max_new_tokens=256 \
  gen.sampling_steps=64 \
  gen.batch_size=4 \
  safety.enabled=true \
  safety.eta=4.0 \
  safety.t_start=0 \
  safety.t_end=18 \
  safety.unsafe_artifact_root=$UNSAFE_ARTIFACT_ROOT \
  io.experiment_slug=quickstart_run
```

Completions are written to `results/quickstart_run/<run_id>/completions.jsonl`.

### 5. Score safety

```bash
export LLAMAGUARD_PATH=./models/LlamaGuard-...

python -m tools.score \
  score.run_dir=results/quickstart_run \
  score.classifier=llamaguard \
  score.classifier_model=$LLAMAGUARD_PATH
```

---

## Models

| Model | HuggingFace |
|---|---|
| LLaDA-8B-Instruct | `GSAI-ML/LLaDA-8B-Instruct` |
| LLaDA-8B-Base | `GSAI-ML/LLaDA-8B-Base` |
| Dream-v0-Instruct-7B | `Dream-org/Dream-v0-Instruct-7B` |
| MDLM | [kuleshov-group/mdlm-owt](https://huggingface.co/kuleshov-group/mdlm-owt) |

For jailbreak evaluation datasets (JBB, HarmBench, AdvBench, StrongREJECT, WildJailbreak):
```bash
python scripts/setup_jailbreak_assets.py --out $JAILBREAK_DATA_ROOT
```

---

## Configuration

All tools are [Hydra](https://hydra.cc/) applications reading `configs/config.yaml`. Every key can be overridden from the CLI.

| Key | What it controls |
|---|---|
| `model.family` | Backend: `llada`, `mdlm`, `dream` |
| `model.checkpoint` / `model.tokenizer_name` | Resolved from `$CHECKPOINT_PATH` / `$TOKENIZER_PATH` |
| `gen.max_new_tokens`, `gen.sampling_steps`, `gen.batch_size` | Generation settings |
| `safety.enabled` / `safety.eta` | Toggle SAD; main guidance strength |
| `safety.t_start` / `safety.t_end` | Guidance window (step indices). For LLaDA at 64 steps, `t_end=18` ≈ final 28%. |
| `safety.unsafe_artifact_root` | Path to pre-built reference tensors |
| `io.experiment_slug` | Output directory name |

To switch datasets, change the `defaults:` entry in `configs/config.yaml`:
```yaml
defaults:
  - data/catalog          # full eval
  - data/catalog_test     # small smoke-test
  - data/catalog_jailbreak  # jailbreak suite
```

---

## Repository Layout

```
assets/                     # Figures for this README
configs/                    # Hydra configs (data catalogs, slurm pipelines, unsafe-prep sweeps)
scripts/                    # Analysis, plotting, and debug utilities
src/
  sampling/                 # Model-agnostic generation layer (backends, safe hooks, sample_text)
  tools/                    # CLI entry points: generate, score, aggregate, hazard_report
  safety_eval/              # Safety classifiers (LlamaGuard, ToxiGen, HarmBench ASR)
  unsafe_prep/              # Build unsafe reference tensors from benchmark datasets
  utils/                    # Shared constants, prompt loading, and RBF utilities
  slurm/                    # Slurm job scripts and Python pipeline submitters
  third_party/              # Vendored upstream repos (see src/third_party/README.md)
    mdlm/                   # MDLM model + repellency/ (our method)
    LLaDA/                  # LLaDA-8B generation code
    Dream/                  # Dream-v0 generation code
    DiffuGuard/             # DiffuGuard jailbreak defense (comparison baseline)
    DIJA/                   # DIJA jailbreak attack (comparison baseline)
    HarmBench/              # HarmBench ASR classifier
```

`third_party/` contains vendored copies of upstream repos included for reproducibility. Do not modify them directly — our additions live in `src/sampling/`, `src/tools/`, and `src/third_party/mdlm/repellency/`.

---

## Running the Full Pipeline (Slurm)

```bash
# Build unsafe reference tensors
python src/slurm/submit_sbatch_unsafe_prep.py \
  --config configs/slurm/unsafe_prep_submit.yaml \
  --repo-root "$REPO_ROOT"

# Generate + score (full sweep)
python src/slurm/submit_sbatch_experiments.py \
  --config configs/slurm/sbatch_prompt_pipeline.yaml \
  --repo-root "$REPO_ROOT"

# Jailbreak evaluation
python src/slurm/submit_sbatch_jailbreak.py \
  --config configs/slurm/sbatch_eval_diffguard_jailbreak_llada_instruct.yaml \
  --repo-root "$REPO_ROOT"
```

See [`README_jailbreak.md`](README_jailbreak.md) for DiffuGuard and DIJA evaluation details, and [`src/tools/README.md`](src/tools/README.md) for all tool options.

---

## Tests

```bash
# Focused: unsafe-prep + safe denoiser
python -m pytest -q src/unsafe_prep/tests \
  src/third_party/mdlm/repellency/tests/test_safe_denoiser_alignment.py

# Full suite (excluding vendored third-party)
python -m pytest -q src --ignore=src/third_party
```

---

## Compute Canada Notes

<details>
<summary>Environment setup</summary>

```bash
pip install -r src/requirements-cc.txt --no-index          # most nodes
pip install -r src/requirements-cc-h100.txt --no-index     # H100 nodes
```

- Flash Attention requires `torch==2.7.1+computecanada` + `flash-attn==2.8.3+computecanada`. Do not use Mamba.
- Replace `rrg-<your-PI>` in sbatch scripts with your actual allocation.
- Log directories (`/scratch/%u/logs/safe-text-diffusion/`) must exist before job submission.

See `scripts/env_profile.sh` for a convenience wrapper that loads modules, activates the venv, and exports all paths.
</details>

<details>
<summary>Slurm debug cheatsheet</summary>

```bash
squeue -j <jobid> -o "%.18i %.2t %.10P %R"   # why is my job pending?
sprio -j <jobid>                               # priority
sshare -l -A <account>                         # fairshare
squeue --start -j <jobid> -o "%.18i %.19S %R"  # estimated start
sbatch --test-only <script.sh>                 # validate without submitting
partition-stats && sq                          # queue overview
```

Flash Attention / torch version fix:
```bash
pip uninstall -y torch flash-attn causal-conv1d torchvision torchaudio torchmetrics timm mamba-ssm
pip install --no-index torch==2.7.1+computecanada
pip install --no-index flash-attn==2.8.3+computecanada
```
</details>

<details>
<summary>Full test workflow (interactive allocation)</summary>

```bash
salloc --account=<your-account> --gpus-per-node=a100:1 \
  --cpus-per-task=1 --mem=8GB --time=2:50:0

module load StdEnv/2023 cuda/12.2 python/3.11 gcc arrow/21.0.0 scipy-stack
source .venv/bin/activate

export REPO_ROOT=~/repos/safe-text-diffusion
export HF_HOME=$SCRATCH/hf_home
export HF_DATASETS_CACHE=$SCRATCH/hf_datasets
export HF_MODELS_CACHE=$SCRATCH/hf_models
export CHECKPOINT_PATH=$SCRATCH/models/text-diffusion/mdlm.ckpt
export TOKENIZER_PATH=$SCRATCH/hf_models/gpt2-large
export PYTHONPATH=$REPO_ROOT:$REPO_ROOT/src:$REPO_ROOT/src/third_party/mdlm:$PYTHONPATH

python -m pytest -q src/unsafe_prep/tests \
  src/third_party/mdlm/repellency/tests/test_safe_denoiser_alignment.py

python src/slurm/submit_sbatch_experiments.py \
  --config configs/slurm/sbatch_submit_test.yaml \
  --repo-root "$REPO_ROOT" --integration-test
```
</details>

---

## Citation

If you find this work useful, please cite:

```bibtex
@misc{yusuf2026safetyawaredenoisertextdiffusion,
      title={The Safety-Aware Denoiser for Text Diffusion Models}, 
      author={Amman Yusuf and Zhejun Jiang and Mijung Park},
      year={2026},
      eprint={2605.08116},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2605.08116}, 
}
```

---

<p align="center">
  <a href="https://arxiv.org/abs/2605.08116">[Paper]</a> &nbsp;|&nbsp;
  <a href="https://github.com/ammanyusuf/SAD">[Code]</a>
</p>
