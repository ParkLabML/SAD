# Safe Denoiser for Discrete Diffusion Language Models

<p align="center">
  <a href="https://arxiv.org/abs/2605.08116">[Paper]</a> &nbsp;|&nbsp;
  <a href="#citation">[Citation]</a> &nbsp;|&nbsp;
  <a href="code/">[Code]</a>
</p>

> **Safe Denoiser for Discrete Diffusion Language Models**
> [Amman Yusuf](ammanyusuf.com), [Zhejun Jiang], [Mijung Park](https://www.cs.ubc.ca/~mijungp/)
> [ICML 2026]

> **Code:** The full implementation is maintained at [`code/`](code/) (submodule → [`ammanyusuf/SAD`](https://github.com/ammanyusuf/SAD)). Clone with `--recurse-submodules` to include it.

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

## Code

Full implementation, quick start, configuration reference, and Slurm pipeline docs are in the [`code/`](code/) submodule ([`ammanyusuf/SAD`](https://github.com/ammanyusuf/SAD)).

```bash
git clone --recurse-submodules https://github.com/ParkLabML/SAD.git
```

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
  <a href="code/">[Code]</a>
</p>
