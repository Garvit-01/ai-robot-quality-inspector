#!/usr/bin/env python3
"""Measure NVIDIA's pretrained Factory policy (e.g. nut threading) and film it close up.

Based on Isaac Lab's scripts/reinforcement_learning/rl_games/play.py, with three changes:
  - it runs a fixed number of episodes and counts successes per episode, from the task's
    own physics check (`ep_succeeded`: the nut seated within 2.5 mm sideways and 0.75 mm in
    height of its target, at any point in the episode) and the task's end-of-episode rate;
  - the recorded video's camera looks at environment 0's bolt from ~30 cm, instead of the
    default 7.5 m, where a 24 mm nut is a few pixels;
  - results go to a JSON file.

Factory's policy is "blind": its inputs are gripper pose/velocity relative to the bolt, no
camera. It threads whatever it is handed, which is why the inspector decides which nuts
reach it.

Runs in Isaac Lab's conda environment (not this project's .venv):
    cd ~/IsaacLab && env -u VIRTUAL_ENV ~/miniconda3/bin/conda run --no-capture-output -n env_isaaclab \\
        ./isaaclab.sh -p ~/nvidia_nebius_challenge_ws/isaac/factory_eval.py \\
        --task Isaac-Factory-NutThread-Direct-v0 --num_envs 16 --episodes 3 --headless \\
        --out ~/nvidia_nebius_challenge_ws/runs/factory/nutthread
"""
import argparse
import json
import math
import os
import sys
import time

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", default="Isaac-Factory-NutThread-Direct-v0")
parser.add_argument("--num_envs", type=int, default=16)
parser.add_argument("--episodes", type=int, default=3, help="episode rounds (all envs reset together)")
parser.add_argument("--out", required=True)
parser.add_argument("--no_video", action="store_true")
parser.add_argument("--eye", type=float, nargs=3, default=[0.82, 0.22, 0.20], help="camera, env-0 frame (m)")
parser.add_argument("--lookat", type=float, nargs=3, default=[0.60, 0.0, 0.06], help="camera target, env-0 frame")
parser.add_argument("--seed", type=int, default=42)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
args_cli.enable_cameras = not args_cli.no_video
sys.argv = [sys.argv[0]] + hydra_args
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from rl_games.common import env_configurations, vecenv  # noqa: E402
from rl_games.torch_runner import Runner  # noqa: E402

import isaaclab_tasks  # noqa: E402,F401
from isaaclab_rl.rl_games import RlGamesGpuEnv, RlGamesVecEnvWrapper  # noqa: E402
from isaaclab_rl.utils.pretrained_checkpoint import get_published_pretrained_checkpoint  # noqa: E402
from isaaclab_tasks.utils.hydra import hydra_task_config  # noqa: E402


@hydra_task_config(args_cli.task, "rl_games_cfg_entry_point")
def main(env_cfg, agent_cfg: dict):
    os.makedirs(args_cli.out, exist_ok=True)
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = agent_cfg["params"]["seed"] = args_cli.seed
    # Film environment 0's bolt from close by.
    env_cfg.viewer.origin_type = "env"
    env_cfg.viewer.env_index = 0
    env_cfg.viewer.eye = tuple(args_cli.eye)
    env_cfg.viewer.lookat = tuple(args_cli.lookat)

    ckpt = get_published_pretrained_checkpoint("rl_games", args_cli.task)
    if not ckpt:
        print("[eval] no pretrained checkpoint for", args_cli.task)
        return
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None if args_cli.no_video else "rgb_array")
    base = env.unwrapped
    steps_per_episode = int(base.max_episode_length)
    if not args_cli.no_video:
        env = gym.wrappers.RecordVideo(env, video_folder=args_cli.out, step_trigger=lambda s: s == 0,
                                       video_length=steps_per_episode, disable_logger=True)
    p = agent_cfg["params"]
    env = RlGamesVecEnvWrapper(env, p["config"]["device"], p["env"].get("clip_observations", math.inf),
                               p["env"].get("clip_actions", math.inf), p["env"].get("obs_groups"),
                               p["env"].get("concate_obs_groups", True))
    vecenv.register("IsaacRlgWrapper", lambda config_name, num_actors, **kw: RlGamesGpuEnv(config_name, num_actors, **kw))
    env_configurations.register("rlgpu", {"vecenv_type": "IsaacRlgWrapper", "env_creator": lambda **kw: env})
    p["load_checkpoint"], p["load_path"] = True, ckpt
    p["config"]["num_actors"] = base.num_envs
    runner = Runner()
    runner.load(agent_cfg)
    agent = runner.create_player()
    agent.restore(ckpt)
    agent.reset()

    obs = env.reset()
    obs = obs["obs"] if isinstance(obs, dict) else obs
    _ = agent.get_batch_size(obs, 1)
    if agent.is_rnn:
        agent.init_rnn()

    print(f"[eval] {args_cli.task}: {base.num_envs} envs x {args_cli.episodes} episodes, "
          f"{steps_per_episode} steps each", flush=True)
    episodes, t0 = [], time.time()
    for ep in range(args_cli.episodes):
        ever = torch.zeros(base.num_envs, dtype=torch.bool, device=base.device)
        end_rate = None
        for t in range(steps_per_episode):
            with torch.inference_mode():
                actions = agent.get_action(agent.obs_to_torch(obs), is_deterministic=agent.is_deterministic)
                obs, _, dones, info = env.step(actions)
                # Inside inference mode, as in NVIDIA's play.py: outside it, the in-place reset of
                # the RNN state raised on the first step (Oct 14).
                if agent.is_rnn and agent.states is not None and len(dones) > 0:
                    for s in agent.states:
                        s[:, dones, :] = 0.0
            if t < steps_per_episode - 1:  # the final step resets the envs, which clears ep_succeeded
                ever |= base.ep_succeeded.bool()
            if isinstance(info, dict) and "successes" in info:
                end_rate = float(info["successes"])
        n_ok = int(ever.sum())
        episodes.append({"episode": ep, "succeeded_at_some_point": n_ok, "envs": base.num_envs,
                         "success_rate_at_end": end_rate})
        print(f"[eval] episode {ep}: {n_ok}/{base.num_envs} succeeded; task's end-of-episode rate {end_rate}",
              flush=True)
    total = sum(e["succeeded_at_some_point"] for e in episodes)
    n = sum(e["envs"] for e in episodes)
    res = {"task": args_cli.task, "checkpoint": ckpt, "num_envs": base.num_envs, "episodes": episodes,
           "success": total, "attempts": n, "success_rate": round(total / n, 3),
           "seconds": round(time.time() - t0, 1), "steps_per_episode": steps_per_episode}
    with open(os.path.join(args_cli.out, "result.json"), "w") as fh:
        json.dump(res, fh, indent=2)
    print(f"[eval] RESULT {total}/{n} = {res['success_rate']:.1%} in {res['seconds']}s -> {args_cli.out}", flush=True)
    env.close()


if __name__ == "__main__":
    # Always close the app: after an exception inside main(), Isaac Sim otherwise stays open,
    # idle, and a background job never ends (it sat for 2 hours on Oct 14).
    try:
        main()
    finally:
        simulation_app.close()
