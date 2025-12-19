import argparse
from flask import Flask, jsonify, render_template, request
import numpy as np
from life_game_env import LifeGameEnv

app = Flask(__name__)

ENV = None
OBS = None
INFO = None


def get_env():
    global ENV
    if ENV is None:
        ENV = LifeGameEnv(action_mode="text", include_social_state=True)
    return ENV


def reset_env():
    global OBS, INFO
    env = get_env()
    OBS, INFO = env.reset()
    payload = dict(INFO)
    payload["step"] = len(getattr(env, "narrative_log", []))
    return _sanitize_payload(payload)


def step_env(action_text: str):
    global OBS, INFO
    env = get_env()
    OBS, reward, terminated, truncated, INFO = env.step(action_text)
    payload = dict(INFO)
    payload["reward"] = reward
    payload["terminated"] = terminated
    payload["truncated"] = truncated
    payload["step"] = len(getattr(env, "narrative_log", []))
    return _sanitize_payload(payload)


def _sanitize_payload(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    if isinstance(obj, dict):
        return {k: _sanitize_payload(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize_payload(v) for v in obj]
    return obj


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/reset", methods=["POST"])
def api_reset():
    info = reset_env()
    return jsonify(info)


@app.route("/api/state", methods=["GET"])
def api_state():
    if INFO is None:
        info = reset_env()
    else:
        info = INFO
    return jsonify(info)


@app.route("/api/metrics", methods=["GET"])
def api_metrics():
    env = get_env()
    payload = {
        "metrics": env.get_breakthrough_metrics(),
        "reward_components": env.reward_components,
        "step": len(getattr(env, "narrative_log", [])),
        "age": getattr(env, "current_age", 0),
        "temporal_consistency": getattr(env, "temporal_consistency_score", 0.0),
    }
    return jsonify(_sanitize_payload(payload))


@app.route("/api/step", methods=["POST"])
def api_step():
    payload = request.get_json(silent=True) or {}
    action_text = payload.get("action_text", "")
    if not isinstance(action_text, str) or not action_text.strip():
        return jsonify({"error": "action_text is required"}), 400
    return jsonify(step_env(action_text))


def main() -> None:
    parser = argparse.ArgumentParser(description="Phillnet Complete Life Studio Web App")
    parser.add_argument("--host", default="127.0.0.1", help="Host interface to bind")
    parser.add_argument("--port", type=int, default=5000, help="Port to listen on")
    parser.add_argument("--debug", action="store_true", help="Enable Flask debug mode")
    parser.add_argument("--no-reset", action="store_true", help="Skip env reset on startup")
    args = parser.parse_args()

    if not args.no_reset:
        reset_env()
    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
