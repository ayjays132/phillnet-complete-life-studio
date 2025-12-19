from typing import Dict, Any, List, Tuple, Callable, Optional

from life_game_env import LifeGameEnv
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch


def make_env(action_mode: str = "text", include_social_state: bool = True) -> LifeGameEnv:
    return LifeGameEnv(action_mode=action_mode, include_social_state=include_social_state)


def run_text_episode(env: LifeGameEnv, policy_fn: Callable, max_steps: int = 50) -> Dict[str, Any]:
    obs, info = env.reset()
    total_reward = 0.0
    steps = 0

    for _ in range(max_steps):
        action_text = policy_fn(obs, info)
        obs, reward, terminated, truncated, info = env.step(action_text)
        total_reward += reward
        steps += 1
        if terminated or truncated:
            break

    return {"total_reward": total_reward, "steps": steps}


def run_text_episode_log(
    env: LifeGameEnv, policy_fn: Callable, max_steps: int = 50
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    obs, info = env.reset()
    total_reward = 0.0
    steps = 0
    log: List[Dict[str, Any]] = []

    for _ in range(max_steps):
        action_text = policy_fn(obs, info)
        obs, reward, terminated, truncated, info = env.step(action_text)
        total_reward += reward
        steps += 1
        log.append(
            {
                "step": steps,
                "scenario_id": info.get("scenario_id"),
                "action_text": info.get("action_text"),
                "action_source": info.get("action_source"),
                "reward": reward,
            }
        )
        if terminated or truncated:
            break

    return {"total_reward": total_reward, "steps": steps}, log


def run_batch(env: LifeGameEnv, policy_fn: Callable, episodes: int = 10, max_steps: int = 50) -> Dict[str, Any]:
    totals = []
    steps_list = []
    for _ in range(episodes):
        stats = run_text_episode(env, policy_fn, max_steps=max_steps)
        totals.append(stats["total_reward"])
        steps_list.append(stats["steps"])
    return {
        "episodes": episodes,
        "mean_reward": float(sum(totals) / len(totals)) if totals else 0.0,
        "mean_steps": float(sum(steps_list) / len(steps_list)) if steps_list else 0.0,
    }


def load_hf_model(
    model_id: str,
    device: Optional[str] = None,
    dtype: Optional[str] = None,
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if dtype == "fp16":
        torch_dtype = torch.float16
    elif dtype == "bf16":
        torch_dtype = torch.bfloat16
    else:
        torch_dtype = torch.float32

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch_dtype)
    model.to(device)
    model.eval()
    return model, tokenizer, device


def make_hf_policy(
    model,
    tokenizer,
    device: str,
    max_new_tokens: int = 24,
    temperature: float = 0.8,
    top_p: float = 0.9,
) -> Callable:
    def policy_fn(obs, info):
        options = info.get("available_options", [])
        context = info.get("scenario_text", "")
        option_lines = "\n".join([f"{i}. {opt}" for i, opt in enumerate(options)])
        prompt = (
            "You are choosing the best action in a life simulation.\n"
            f"Context: {context}\n"
            "Options:\n"
            f"{option_lines}\n"
            "Answer with a short action description or the option text.\n"
            "Choice:"
        )
        inputs = tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                pad_token_id=tokenizer.eos_token_id,
            )
        generated = tokenizer.decode(
            output_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        )
        return generated.strip()

    return policy_fn
