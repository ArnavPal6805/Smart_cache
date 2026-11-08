from __future__ import annotations

import argparse
import random
from pathlib import Path
from statistics import mean
from time import perf_counter_ns

import numpy as np
import torch

from smartcache.environment.byte_cache_env import ByteCacheEnv
from smartcache.environment.workload import generate_workload
from smartcache.training.dqn_agent import DQNAgent
from smartcache.training.replay_buffer import ReplayBuffer


def evaluate(
    agent: DQNAgent,
    capacity_bytes: int,
    max_objects: int,
    requests_per_trace: int,
    seeds: list[int],
    frequency_window: int,
    recency_window: int,
    cost_scale_ms: float,
    size_scale_bytes: float,
) -> dict[str, float]:
    costs: list[float] = []
    hit_ratios: list[float] = []
    decision_times_ns: list[int] = []
    decisions_per_trace: list[int] = []

    for seed in seeds:
        requests = generate_workload(
            number_of_requests=requests_per_trace,
            seed=seed,
            drift=True,
        )
        env = ByteCacheEnv(
            requests,
            capacity_bytes=capacity_bytes,
            max_objects=max_objects,
            frequency_window=frequency_window,
            recency_window=recency_window,
            cost_scale_ms=cost_scale_ms,
            size_scale_bytes=size_scale_bytes,
        )
        obs, _ = env.reset(seed=seed)
        total_cost = 0.0
        completed_requests = 0
        hits = 0
        decisions = 0
        while True:
            mask = env.action_mask()
            if mask[:max_objects].any():
                start = perf_counter_ns()
                action = agent.act(obs, epsilon=0.0, action_mask=mask)
                decision_times_ns.append(perf_counter_ns() - start)
                decisions += 1
            else:
                action = env.noop_action
            obs, _, terminated, truncated, info = env.step(action)
            total_cost += info["miss_cost_ms"]
            if info["request_complete"]:
                completed_requests += 1
                hits += int(info["hit"])
            if terminated or truncated:
                break
        costs.append(total_cost)
        hit_ratios.append(hits / max(1, completed_requests))
        decisions_per_trace.append(decisions)

    return {
        "miss_cost_ms": mean(costs),
        "hit_ratio": mean(hit_ratios),
        "decision_time_ms": mean(decision_times_ns) / 1_000_000
        if decision_times_ns
        else 0.0,
        "decisions_per_trace": mean(decisions_per_trace),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--requests", type=int, default=1000)
    parser.add_argument("--capacity-bytes", type=int, default=24576)
    parser.add_argument("--max-objects", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--cost-scale-ms", type=float, default=250.0)
    parser.add_argument("--size-scale-bytes", type=float, default=32768.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    frequency_window = 100
    recency_window = 1000
    agent = DQNAgent(
        state_dim=args.max_objects * 5 + 5,
        action_dim=args.max_objects + 1,
        gamma=1.0,
        device=args.device,
    )
    replay = ReplayBuffer()
    best_cost = float("inf")

    for episode in range(args.episodes):
        requests = generate_workload(
            number_of_requests=args.requests,
            seed=args.seed + episode,
            drift=True,
        )
        env = ByteCacheEnv(
            requests,
            capacity_bytes=args.capacity_bytes,
            max_objects=args.max_objects,
            frequency_window=frequency_window,
            recency_window=recency_window,
            cost_scale_ms=args.cost_scale_ms,
            size_scale_bytes=args.size_scale_bytes,
        )
        obs, _ = env.reset(seed=args.seed + episode)
        epsilon = max(0.05, 1.0 - episode / max(1, args.episodes - 1))
        losses: list[float] = []
        total_cost = 0.0

        while True:
            mask = env.action_mask()
            if mask[:args.max_objects].any():
                action = agent.act(obs, epsilon=epsilon, action_mask=mask)
            else:
                action = env.noop_action
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            replay.add(obs, action, reward, next_obs, done, env.action_mask())
            obs = next_obs
            total_cost += info["miss_cost_ms"]
            if len(replay) >= args.batch_size:
                losses.append(
                    agent.learn(
                        replay.sample(args.batch_size, agent.device),
                        reward_scale_ms=agent.reward_scale_ms,
                    )
                )
            if done:
                break

        validation = evaluate(
            agent,
            args.capacity_bytes,
            args.max_objects,
            args.requests,
            [args.seed + 100_000, args.seed + 200_000],
            frequency_window,
            recency_window,
            args.cost_scale_ms,
            args.size_scale_bytes,
        )
        metadata = {
            "capacity_bytes": args.capacity_bytes,
            "max_objects": args.max_objects,
            "frequency_window": frequency_window,
            "recency_window": recency_window,
            "episode": episode + 1,
            "feature_version": "byte_cache_v2",
            "cost_scale_ms": args.cost_scale_ms,
            "size_scale_bytes": args.size_scale_bytes,
        }
        agent.save_checkpoint("artifacts/byte_dqn_latest.pt", metadata)
        if validation["miss_cost_ms"] < best_cost:
            best_cost = validation["miss_cost_ms"]
            agent.save_checkpoint(
                "artifacts/byte_dqn_best.pt",
                metadata | validation,
            )
        print(
            f"episode={episode + 1} cost_ms={total_cost:.1f} "
            f"loss={mean(losses) if losses else 0.0:.5f} "
            f"validation_cost_ms={validation['miss_cost_ms']:.1f}"
        )


if __name__ == "__main__":
    main()
