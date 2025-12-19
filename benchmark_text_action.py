import argparse
import json
import time
from typing import Dict, Any, Tuple, List

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from life_game_env import LifeGameEnv


def build_prompt(context: str, options: list) -> str:
    opt_lines = "\n".join([f"{i}. {opt}" for i, opt in enumerate(options)])
    return (
        "You are choosing the best action in a life simulation.\n"
        f"Context: {context}\n"
        "Options:\n"
        f"{opt_lines}\n"
        "Answer with a short action description or the option text.\n"
        "Choice:"
    )


def generate_action_text(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 24,
) -> str:
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.8,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
        )
    generated = tokenizer.decode(output_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return generated.strip()


def run_episode(
    env: LifeGameEnv,
    model,
    tokenizer,
    mode: str,
    max_steps: int,
    log_steps: bool,
) -> Dict[str, Any]:
    obs, info = env.reset()
    total_reward = 0.0
    invalid_actions = 0
    action_sources = []
    match_scores = []
    steps: List[Dict[str, Any]] = []

    for _ in range(max_steps):
        options = info.get("available_options", [])
        context = info.get("scenario_text", "")

        if mode == "index":
            action = env.action_space.sample()
            model_output = None
        else:
            prompt = build_prompt(context, options)
            action_text = generate_action_text(model, tokenizer, prompt)
            action = action_text
            model_output = action_text

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        if info.get("action_match_score", 1.0) <= 0.0:
            invalid_actions += 1
        if "action_source" in info:
            action_sources.append(info["action_source"])
        if "action_match_score" in info:
            match_scores.append(float(info["action_match_score"]))
        if log_steps:
            steps.append(
                {
                    "scenario_id": info.get("scenario_id"),
                    "age": info.get("age"),
                    "stage": info.get("stage"),
                    "scenario_text": info.get("scenario_text"),
                    "available_options": info.get("available_options", []),
                    "model_output": model_output,
                    "action_index": info.get("action_index"),
                    "action_text": info.get("action_text"),
                    "action_text_generalized": info.get("action_text_generalized"),
                    "action_source": info.get("action_source"),
                    "action_match_score": info.get("action_match_score"),
                    "reward": reward,
                }
            )

        if terminated or truncated:
            break

    return {
        "total_reward": total_reward,
        "invalid_actions": invalid_actions,
        "mean_match_score": float(np.mean(match_scores)) if match_scores else None,
        "action_sources": action_sources,
        "steps": steps,
    }


def run_benchmark(episodes: int, max_steps: int, log_steps: bool) -> Dict[str, Any]:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    model = AutoModelForCausalLM.from_pretrained("gpt2", torch_dtype=dtype)
    model.to(device)
    model.eval()

    results = {}
    for mode in ["index", "text"]:
        env = LifeGameEnv(action_mode=mode)
        episode_rewards = []
        invalid_counts = []
        match_scores = []
        action_source_counts: Dict[str, int] = {}

        episodes_log = []
        for _ in range(episodes):
            stats = run_episode(env, model, tokenizer, mode, max_steps, log_steps)
            episode_rewards.append(stats["total_reward"])
            invalid_counts.append(stats["invalid_actions"])
            if stats["mean_match_score"] is not None:
                match_scores.append(stats["mean_match_score"])
            for src in stats["action_sources"]:
                action_source_counts[src] = action_source_counts.get(src, 0) + 1
            if log_steps:
                episodes_log.append(
                    {
                        "total_reward": stats["total_reward"],
                        "invalid_actions": stats["invalid_actions"],
                        "mean_match_score": stats["mean_match_score"],
                        "steps": stats["steps"],
                    }
                )

        results[mode] = {
            "episodes": episodes,
            "max_steps": max_steps,
            "mean_reward": float(np.mean(episode_rewards)),
            "std_reward": float(np.std(episode_rewards)),
            "mean_invalid_actions": float(np.mean(invalid_counts)),
            "mean_match_score": float(np.mean(match_scores)) if match_scores else None,
            "action_source_counts": action_source_counts,
            "device": device,
            "dtype": str(dtype),
        }
        if log_steps:
            results[mode]["episodes_log"] = episodes_log

    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=30)
    parser.add_argument("--out", type=str, default="benchmark_results.json")
    parser.add_argument("--log-steps", action="store_true")
    args = parser.parse_args()

    start = time.time()
    results = run_benchmark(args.episodes, args.max_steps, args.log_steps)
    results["elapsed_sec"] = round(time.time() - start, 2)

    with open(args.out, "w", encoding="ascii") as f:
        json.dump(results, f, indent=2)

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
