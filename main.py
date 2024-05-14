import argparse
import warnings

import numpy as np
import torch

# get configs

from config import args_cheetah_run, args_cartpole_swingup, args_finger_spin, args_cup_catch, args_fish_upright, args_reacher_easy, \
    args_walker_walk, args_hopper_stand, args_quadruped_walk, args_swimmer_swimmer6, args_reacher_hard, args_swimmer_swimmer15
    
from environments.parallel_envs import make_vec_envs
from learner import Learner


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env-type', default='cartpole_swingup')
    args, rest_args = parser.parse_known_args()
    env = args.env_type
    
    # load env config
    if env == 'cheetah_run':
        args = args_cheetah_run.get_args(rest_args)
    elif env == 'cartpole_swingup':
        args = args_cartpole_swingup.get_args(rest_args)
    elif env == 'finger_spin':
        args = args_finger_spin.get_args(rest_args)
    elif env == 'cup_catch':
        args = args_cup_catch.get_args(rest_args)
    elif env == 'fish_upright':
        args = args_fish_upright.get_args(rest_args)
    elif env == 'reacher_easy':
        args = args_reacher_easy.get_args(rest_args)
    elif env == 'walker_walk':
        args = args_walker_walk.get_args(rest_args)
    elif env == 'hopper_stand':
        args = args_hopper_stand.get_args(rest_args)
    elif env == 'swimmer_swimmer6':
        args = args_swimmer_swimmer6.get_args(rest_args)
    elif env == 'quadruped_walk':
        args = args_quadruped_walk.get_args(rest_args)
    elif env == 'reacher_hard':
        args = args_reacher_hard.get_args(rest_args)
    elif env == 'swimmer_swimmer15':
        args = args_swimmer_swimmer15.get_args(rest_args)
    
    else:
        raise Exception("Invalid Environment")

    if args.deterministic_execution:
        print('Envoking deterministic code execution.')
        if torch.backends.cudnn.enabled:
            warnings.warn('Running with deterministic CUDNN.')
        if args.num_processes > 1:
            raise RuntimeError('If you want fully deterministic code, run it with num_processes=1.'
                               'Warning: This will slow things down and might break A2C if '
                               'policy_num_steps < env._max_episode_steps.')

    if args.norm_actions_pre_sampling or args.norm_actions_post_sampling:
        envs = make_vec_envs(env_name=args.env_name, seed=0, num_processes=args.num_processes,
                             gamma=args.policy_gamma, device='cpu',
                             episodes_per_task=args.max_rollouts_per_task,
                             normalise_rew=args.norm_rew_for_policy, ret_rms=None,
                             tasks=None,
                             )
        assert np.unique(envs.action_space.low) == [-1]
        assert np.unique(envs.action_space.high) == [1]

    if args.disable_metalearner or args.disable_decoder:
        args.decode_reward = False
        args.decode_state = False
        args.decode_task = False

    if hasattr(args, 'decode_only_past') and args.decode_only_past:
        args.split_batches_by_elbo = True

    seed_list = [args.seed] if isinstance(args.seed, int) else args.seed
    for seed in seed_list:
        print('training', seed)
        args.seed = seed
        args.action_space = None

        learner = Learner(args)
        learner.train()


if __name__ == '__main__':
    main()
