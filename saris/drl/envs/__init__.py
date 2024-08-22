from gymnasium.envs.registration import register


def register_envs():
    register(
        id="wireless-sigmap-v0",
        entry_point="saris.drl.envs.wireless:WirelessEnvV0",
        max_episode_steps=250,
    )
