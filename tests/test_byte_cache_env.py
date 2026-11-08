from smartcache.environment.byte_cache_env import ByteCacheEnv


def test_multistep_eviction_charges_miss_once():
    requests = [
        {"key": "a", "cost": 1.0, "size": 40},
        {"key": "b", "cost": 2.0, "size": 40},
        {"key": "c", "cost": 3.0, "size": 70},
    ]
    env = ByteCacheEnv(requests, capacity_bytes=100, max_objects=4)
    _, _ = env.reset()

    for expected_reward in (-1.0, -2.0):
        _, reward, terminated, _, info = env.step(env.noop_action)
        assert reward == expected_reward
        assert info["request_complete"]
        assert not terminated

    assert env.action_mask()[:2].all()
    _, reward, terminated, _, info = env.step(0)
    assert reward == -3.0
    assert not info["request_complete"]
    assert not terminated
    _, reward, terminated, _, info = env.step(0)
    assert reward == 0.0
    assert info["request_complete"]
    assert terminated


def test_oversized_request_bypasses_without_eviction():
    env = ByteCacheEnv(
        [
            {"key": "small", "cost": 2.0, "size": 20},
            {"key": "large", "cost": 80.0, "size": 101},
        ],
        capacity_bytes=100,
        max_objects=4,
    )
    env.reset()
    env.step(env.noop_action)
    _, reward, terminated, _, info = env.step(env.noop_action)
    assert reward == -80.0
    assert terminated
    assert info["request_complete"]
    assert env.used_bytes == 20
