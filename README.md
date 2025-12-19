# Phillnet Complete Life Studio Gym Environment

Unified task-planning GUI + Gym environment for life simulation training.

## What this is
- A Gymnasium environment built on the Phillnet CompleteLife dataset.
- A production-style GUI that plans tasks and feeds text actions into the env.
- A benchmarking script for text vs index action performance.
- A unified interface for planning-first agents feeding a main policy.

## Install
1) Install locally:
```bash
pip install -e .
```

2) Or install dependencies only:
```bash
pip install -r requirements.txt
```

## Quick start
1) Run the GUI:
```bash
python gui_task_planner.py
```

2) Run the text benchmark:
```bash
python benchmark_text_action.py --episodes 10 --max-steps 50 --log-steps
```

3) Run the web studio:
```bash
python web_app.py
```

4) Use the env in training code:
```python
from training_utils import (
    make_env,
    run_text_episode,
    run_text_episode_log,
    run_batch,
    load_hf_model,
    make_hf_policy,
)

def policy_fn(obs, info):
    # simple baseline: return a short text action
    return "choose the most helpful option"

env = make_env(action_mode="text")
stats = run_text_episode(env, policy_fn, max_steps=25)
print(stats)

stats, log = run_text_episode_log(env, policy_fn, max_steps=10)
print(log[:2])

batch = run_batch(env, policy_fn, episodes=5, max_steps=25)
print(batch)

# Hugging Face model policy
model, tokenizer, device = load_hf_model("gpt2", dtype="fp16")
hf_policy = make_hf_policy(model, tokenizer, device)
stats = run_text_episode(env, hf_policy, max_steps=10)
print(stats)
```

## CLI
After `pip install -e .`:
```bash
phillnet-studio-gui
phillnet-studio-benchmark --episodes 10 --max-steps 50 --log-steps
phillnet-studio-web
```

## Advanced CLI
Web studio options:
```bash
phillnet-studio-web --host 0.0.0.0 --port 8080
phillnet-studio-web --debug
phillnet-studio-web --no-reset
```

Benchmark options:
```bash
phillnet-studio-benchmark --episodes 25 --max-steps 100 --log-steps --out results.json
```

## API (Web Studio)
Endpoints:
- `POST /api/reset` -> reset the env
- `GET /api/state` -> current scenario + state
- `POST /api/step` -> `{ "action_text": "..." }`

Example:
```bash
curl -X POST http://127.0.0.1:5000/api/step \
  -H "Content-Type: application/json" \
  -d "{\"action_text\":\"help the friend\"}"
```

## Production notes
- The web app runs a single in-process env by default. For multi-user use, run multiple instances.
- Use `--host 0.0.0.0` for LAN access and a reverse proxy for public deployments.

## Project layout
- `life_game_env.py` Gym environment
- `gui_task_planner.py` Planner GUI (task + robotics context)
- `benchmark_text_action.py` Text-action benchmark
- `benchmark_results.json` Example benchmark output
- `pyproject.toml` Packaging and CLI entrypoints
- `web_app.py` Flask web app
- `training_utils.py` Minimal helpers for training scripts
- `templates/` Web templates
- `static/` Web assets

## Dataset
This project uses the Phillnet CompleteLife dataset:
`ayjays132/Phillnet-CompleteLife`

Creator: Phillnet Dataset Creator

## Notes
- GUI is optional; the environment works headless.
- Text actions are resolved to options using similarity + numeric hints.
- Designed for planning-first workflows before a main policy.
