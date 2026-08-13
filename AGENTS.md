# Repository Guidelines

## Project Structure & Module Organization

NaVILA is a Python research repository combining training, evaluation, and video/keyframe tooling. Core model and training code lives in `llava/`; benchmark runners and Habitat/VLN-CE integrations are under `evaluation/`. Video preprocessing is in `keyframe/` and `scripts/`, while `configs/` contains environment and extraction YAML files. `VideoMamba/`, `bert/`, and root-level diagnostic scripts provide supporting components and experiments. Images and GIFs are in `assets/`; large datasets, checkpoints, and Habitat scene files should remain external (see `README.md`).

## Build, Test, and Development Commands

- `./environment_setup.sh navila` creates the documented Python 3.10 Conda environment and installs the editable package plus extras. Activate it with `conda activate navila`.
- `pip install -e .` installs the package for local development; use `pip install -e ".[train]"` or `pip install -e ".[eval]"` for optional workflows.
- `pytest` runs the repository's Python tests. GPU- or dataset-dependent checks may require the configured CUDA environment and downloaded data.
- `pre-commit run --all-files` applies/checks Black, isort, Markdown/YAML formatting, and repository hygiene hooks.
- `cd evaluation && bash scripts/eval/r2r.sh CKPT_PATH NUM_CHUNKS CHUNK_START_IDX "GPU_IDS"` runs R2R evaluation; aggregate outputs with `python scripts/eval_jsons.py ./eval_out/CKPT_NAME/VLN-CE-v1/val_unseen NUM_CHUNKS`.

## Coding Style & Naming Conventions

Use Python 3.10-compatible code, four-space indentation, and `snake_case` names for modules, functions, and variables; use `PascalCase` for classes. Black and isort use a 120-character line length. Keep configuration in YAML or existing argument patterns, and avoid committing generated outputs, credentials, model weights, or local dataset paths.

## Testing Guidelines

Add focused tests beside the relevant module, using `test_*.py` or `*_test.py` names and pytest assertions. Run targeted tests while iterating (for example, `pytest llava/finetune_test.py`) and run the full suite before a pull request when dependencies and hardware permit. Record required checkpoints, datasets, GPU count, and any skipped integration checks.

## Commit & Pull Request Guidelines

The history currently contains one concise imperative commit (`first commit`), so use a similarly short subject (for example, `add keyframe sampling test`). Pull requests should explain motivation and affected workflows, identify required data/checkpoints and validation commands, link issues, and include representative logs or screenshots/GIFs for evaluation or visualization changes. Keep unrelated formatting or generated files out.

## Security & Configuration Tips

Do not commit access tokens, private dataset URLs, or credentials. Review shell scripts and config paths before running them, especially commands that install CUDA/FlashAttention or modify site-packages. Treat downloaded checkpoints and evaluation outputs as local artifacts unless explicitly requested for release.
