from evo_gym.envs.registration import registry, register, make, spec


register(
    id='CartPole-survival-v0',
    entry_point='evo_gym.envs.classic_control:CartPoleEnv',
    max_episode_steps=1000000000,
)

register(
    id='ReacherForager-v0',
    entry_point='evo_gym.envs.mujoco.reacherForager:ReacherForagerEnv',
    max_episode_steps=1000000000,
)

register(
    id='HalfCheetahForager-v0',
    entry_point='evo_gym.envs.mujoco.half_cheetah_forager:HalfCheetahForagerEnv',
    max_episode_steps=1000000000,
)

register(
    id='HumanoidForager-v0',
    entry_point='evo_gym.envs.mujoco.humanoid_forager:HumanoidForagerEnv',
    max_episode_steps=1000000000,
)
register(
    id='HumanoidHomeostasis-v0',
    entry_point='evo_gym.envs.mujoco.humanoid_homeostasis:HumanoidHomeostasisEnv',
    max_episode_steps=10000000,
)
register(
    id='WalkerForager-v0',
    entry_point='evo_gym.envs.mujoco.walker_forager:WalkerForagerEnv',
    max_episode_steps=1000000000,
)

register(
    id='Boxing-survival-avoidance-v0',
    entry_point='evo_gym.envs.atari.boxing_survival:BoxingSurvivalAvoidance',
    max_episode_steps=1000000000,
)

register(
    id='Boxing-survival-fight-v0',
    entry_point='evo_gym.envs.atari.boxing_survival:BoxingSurvivalFight',
    max_episode_steps=1000000000,
)

register(
    id='Pong-survival-v0',
    entry_point='evo_gym.envs.atari.pong_survival:PongSurvival',
    max_episode_steps=1000000000,
)

register(
    id='Pong-survival-forager-v0',
    entry_point='evo_gym.envs.atari.pong_survival:PongSurvivalForager',
    max_episode_steps=1000000000,
)

register(
    id='SpaceInvaders-survival-v0',
    entry_point='evo_gym.envs.atari.space_invaders_survival:SpaceInvadersSurvival',
    max_episode_steps=1000000000,
)

register(
    id='BattleZone-survival-forager-v0',
    entry_point='evo_gym.envs.atari.battlezone_survival:BattleZoneSurvivalForager',
    max_episode_steps=1000000000,
)
# Atari
# ----------------------------------------

# # print ', '.join(["'{}'".format(name.split('.')[0]) for name in atari_py.list_games()])
for game in ['adventure', 'air_raid', 'alien', 'amidar', 'assault', 'asterix', 'asteroids', 'atlantis',
    'bank_heist', 'battle_zone', 'beam_rider', 'berzerk', 'bowling', 'boxing', 'breakout', 'carnival',
    'centipede', 'chopper_command', 'crazy_climber', 'defender', 'demon_attack', 'double_dunk',
    'elevator_action', 'enduro', 'fishing_derby', 'freeway', 'frostbite', 'gopher', 'gravitar',
    'hero', 'ice_hockey', 'jamesbond', 'journey_escape', 'kangaroo', 'krull', 'kung_fu_master',
    'montezuma_revenge', 'ms_pacman', 'name_this_game', 'phoenix', 'pitfall', 'pong', 'pooyan',
    'private_eye', 'qbert', 'riverraid', 'road_runner', 'robotank', 'seaquest', 'skiing',
    'solaris', 'space_invaders', 'star_gunner', 'tennis', 'time_pilot', 'tutankham', 'up_n_down',
    'venture', 'video_pinball', 'wizard_of_wor', 'yars_revenge', 'zaxxon']:
    for obs_type in ['image', 'ram']:
        # space_invaders should yield SpaceInvaders-v0 and SpaceInvaders-ram-v0
        name = ''.join([g.capitalize() for g in game.split('_')])
        if obs_type == 'ram':
            name = '{}-ram'.format(name)

        nondeterministic = False
        if game == 'elevator_action' and obs_type == 'ram':
            # ElevatorAction-ram-v0 seems to yield slightly
            # non-deterministic observations about 10% of the time. We
            # should track this down eventually, but for now we just
            # mark it as nondeterministic.
            nondeterministic = True

        register(
            id='{}-v0'.format(name),
            entry_point='evo_gym.envs.atari:AtariEnv',
            kwargs={'game': game, 'obs_type': obs_type, 'repeat_action_probability': 0.25},
            max_episode_steps=10000000,
            nondeterministic=nondeterministic,
        )

        register(
            id='{}-v4'.format(name),
            entry_point='evo_gym.envs.atari:AtariEnv',
            kwargs={'game': game, 'obs_type': obs_type},
            max_episode_steps=100000,
            nondeterministic=nondeterministic,
        )

        # Standard Deterministic (as in the original DeepMind paper)
        if game == 'space_invaders':
            frameskip = 3
        else:
            frameskip = 4

        # Use a deterministic frame skip.
        register(
            id='{}Deterministic-v0'.format(name),
            entry_point='evo_gym.envs.atari:AtariEnv',
            kwargs={'game': game, 'obs_type': obs_type, 'frameskip': frameskip, 'repeat_action_probability': 0.25},
            max_episode_steps=100000,
            nondeterministic=nondeterministic,
        )

        register(
            id='{}Deterministic-v4'.format(name),
            entry_point='evo_gym.envs.atari:AtariEnv',
            kwargs={'game': game, 'obs_type': obs_type, 'frameskip': frameskip},
            max_episode_steps=100000,
            nondeterministic=nondeterministic,
        )

        register(
            id='{}NoFrameskip-v0'.format(name),
            entry_point='evo_gym.envs.atari:AtariEnv',
            kwargs={'game': game, 'obs_type': obs_type, 'frameskip': 1, 'repeat_action_probability': 0.25}, # A frameskip of 1 means we get every frame
            max_episode_steps=frameskip * 100000,
            nondeterministic=nondeterministic,
        )

        # No frameskip. (Atari has no entropy source, so these are
        # deterministic environments.)
        register(
            id='{}NoFrameskip-v4'.format(name),
            entry_point='evo_gym.envs.atari:AtariEnv',
            kwargs={'game': game, 'obs_type': obs_type, 'frameskip': 1}, # A frameskip of 1 means we get every frame
            max_episode_steps=frameskip * 100000,
            nondeterministic=nondeterministic,
        )


# Unit test
# ---------

register(
    id='CubeCrash-v0',
    entry_point='evo_gym.envs.unittest:CubeCrash',
    reward_threshold=0.9,
    )
register(
    id='CubeCrashSparse-v0',
    entry_point='evo_gym.envs.unittest:CubeCrashSparse',
    reward_threshold=0.9,
    )
register(
    id='CubeCrashScreenBecomesBlack-v0',
    entry_point='evo_gym.envs.unittest:CubeCrashScreenBecomesBlack',
    reward_threshold=0.9,
    )

register(
    id='MemorizeDigits-v0',
    entry_point='evo_gym.envs.unittest:MemorizeDigits',
    reward_threshold=20,
    )
