from gymnasium.envs.registration import register


def register_envs():
    register(
        id="wireless-sigmap-v0",
        entry_point="saris.drl.envs.wireless:WirelessEnvV0",
        max_episode_steps=100,
    )

    register(
        id="focal-v0",
        entry_point="saris.drl.envs.focal:WirelessFocalEnvV0",
        max_episode_steps=250,
    )

    register(
        id="wireless-moving-v0",
        entry_point="saris.drl.envs.wireless_moving:WirelessMovingV0",
        max_episode_steps=100,
    )
