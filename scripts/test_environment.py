from smartcache.environment.cache_env import CacheEnv


requests = [
    {"key": "metadata", "cost": 2.0, "size": 1024},
    {"key": "aggregation", "cost": 180.0, "size": 8192},
    {"key": "metadata", "cost": 2.0, "size": 1024},
    {"key": "embedding", "cost": 250.0, "size": 16384},
    {"key": "aggregation", "cost": 180.0, "size": 8192},
    {"key": "metadata", "cost": 2.0, "size": 1024},
]

env = CacheEnv(requests, capacity=2)
obs, info = env.reset(seed=42)
print("Observation shape:", obs.shape)
print("Action space:", env.action_space)
done = False
while not done:
    # Temporary baseline action. The first free slot is filled automatically.
    action = env.noop_action if not env.action_mask()[: env.capacity].any() else 0
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    print(
        f"Request={info['key']:<12} "
        f"Hit={info['hit']!s:<5} "
        f"Reward={reward:>7.1f} "
        f"Evicted={info['evicted']} "
        f"Cache={info['cache']}"
    )
