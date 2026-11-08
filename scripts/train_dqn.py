from __future__ import annotations

import argparse
import random
from pathlib import Path
from statistics import mean
from time import perf_counter_ns

import numpy as np
import torch

from smartcache.environment.cache_env import CacheEnv
from smartcache.environment.workload import generate_workload
from smartcache.training.dqn_agent import DQNAgent
from smartcache.training.replay_buffer import ReplayBuffer


def needs_eviction(env: CacheEnv) -> bool:
    request = env.requests[env.position]
    return str(request["key"]) not in env.cache and None not in env.cache


def evaluate(
    agent: DQNAgent,
    capacity: int,
    requests_per_trace: int,
    seeds: list[int],
    frequency_window: int,
    recency_window: int,
) -> dict[str, float]:
    costs: list[float] = []
    hit_ratios: list[float] = []
    inference_times_ms: list[float] = []
    decision_counts = 0
    for seed in seeds:
        requests = generate_workload(requests_per_trace, seed=seed, drift=True)
        env = CacheEnv(
            requests, capacity=capacity, frequency_window=frequency_window
        )
        env.recency_window = recency_window
        state, _ = env.reset(seed=seed)
        misses = 0
        miss_cost = 0.0
        steps = 0
        while True:
            if needs_eviction(env):
                start = perf_counter_ns()
                action = agent.act(
                    state,
                    epsilon=0.0,
                    action_mask=env.action_mask(),
                )
                inference_times_ms.append((perf_counter_ns() - start) / 1_000_000)
                decision_counts += 1
            else:
                action = env.noop_action
            state, reward, terminated, truncated, info = env.step(action)
            misses += int(not info["hit"])
            miss_cost += info["miss_cost_ms"]
            steps += 1
            if terminated or truncated:
                break
        costs.append(miss_cost)
        hit_ratios.append(1.0 - misses / steps)
    return {
        "miss_cost_ms": mean(costs),
        "hit_ratio": mean(hit_ratios),
        "inference_ms": mean(inference_times_ms) if inference_times_ms else 0.0,
        "decisions": float(decision_counts),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=80)
    parser.add_argument("--requests", type=int, default=800)
    parser.add_argument("--capacity", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    frequency_window = 100
    recency_window = 1000
    state_dim = args.capacity * 5 + 3
    initial_requests = generate_workload(1, seed=args.seed, drift=True)
    initial_env = CacheEnv(
        initial_requests,
        capacity=args.capacity,
        frequency_window=frequency_window,
    )
    agent = DQNAgent(
        state_dim,
        initial_env.action_space.n,
        device=args.device,
    )
    replay = ReplayBuffer()
    best_cost = float("inf")
    artifacts = Path("artifacts")

    for episode in range(args.episodes):
        requests = generate_workload(
            args.requests, seed=args.seed + episode, drift=True
        )
        env = CacheEnv(
            requests, capacity=args.capacity, frequency_window=frequency_window
        )
        env.recency_window = recency_window
        state, _ = env.reset(seed=args.seed + episode)
        epsilon = max(0.05, 1.0 - episode / max(1, args.episodes - 1))
        losses: list[float] = []
        total_cost = 0.0
        while True:
            current_mask = env.action_mask()
            if current_mask[: env.capacity].any():
                action = agent.act(
                    state,
                    epsilon=epsilon,
                    action_mask=current_mask,
                )
            else:
                action = env.noop_action
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            next_mask = env.action_mask()
            replay.add(state, action, reward, next_state, done, next_mask)
            state = next_state
            total_cost += info["miss_cost_ms"]
            if len(replay) >= args.batch_size:
                losses.append(
                    agent.learn(
                        replay.sample(args.batch_size, agent.device),
                        reward_scale_ms=agent.reward_scale_ms,
                    )
                )
            if terminated or truncated:
                break

        metadata = {
            "capacity": args.capacity,
            "frequency_window": frequency_window,
            "recency_window": recency_window,
            "episode": episode + 1,
        }
        agent.save_checkpoint(artifacts / "dqn_latest.pt", metadata)
        validation = evaluate(
            agent,
            args.capacity,
            args.requests,
            [args.seed + 10_000, args.seed + 20_000],
            frequency_window,
            recency_window,
        )
        if validation["miss_cost_ms"] < best_cost:
            best_cost = validation["miss_cost_ms"]
            agent.save_checkpoint(artifacts / "dqn_best.pt", metadata | validation)
        print(
            f"episode={episode + 1} cost_ms={total_cost:.1f} "
            f"loss={mean(losses) if losses else 0.0:.5f} "
            f"validation_cost_ms={validation['miss_cost_ms']:.1f}"
        )


if __name__ == "__main__":
    main()
