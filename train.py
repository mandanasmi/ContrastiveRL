import os
from collections import deque
import torch
import matplotlib.pyplot as plt
import json
import numpy as np
from utils import batch_env
from envs.pycolab import env as pycolab_env
from utils.log import Logger
from algo.conspec.conspec import ConSpec
from algo.ppo import PPO
from utils.log import cleanup_log_dir, update_linear_schedule
from algo.ppo import gail
from nets.policy import Policy
from utils.replay_buffer import ReplayBuffer
import tensorflow.compat.v1 as tf
from six.moves import range
import pickle
import wandb
import datetime


def main(args):
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

     # Create the folder for results 
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    subfolder_name = f"{timestamp}_{args.pycolab_game}_{args.seed}_{args.num_episodes}"
    base_directory = args.save_dir
    full_path = os.path.join(base_directory, subfolder_name)
    os.makedirs(full_path, exist_ok=True)

    if args.cuda and torch.cuda.is_available() and args.cuda_deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    wandb_project_name = 'conspec_v2'
    proj_name = str(args.pycolab_game) +' seed'+ str(args.seed)+ ' episodes'+ str(args.num_episodes)
    logger = Logger(
        exp_name=proj_name,
        save_dir='/home/mila/s/samieima/scratch/conspec_v2',
        print_every=1,
        save_every=1000,
        total_step=1000,
        print_to_stdout=True,
        wandb_project_name=wandb_project_name,
        wandb_tags=['multikeydoor'],
        wandb_config=args,
    )

    log_dir = os.path.expanduser(args.log_dir)
    eval_log_dir = log_dir + "_eval"
    cleanup_log_dir(log_dir)
    cleanup_log_dir(eval_log_dir)

    torch.set_num_threads(1)
    device = torch.device("cuda:0" if args.cuda else "cpu")

    env_builder = pycolab_env.PycolabEnvironment
    env_kwargs = {
        'game': args.pycolab_game, # key to door 4
        'num_apples': args.pycolab_num_apples, # 10 
        'apple_reward': [args.pycolab_apple_reward_min, #0
                         args.pycolab_apple_reward_max], #0
        'fix_apple_reward_in_episode': args.pycolab_fix_apple_reward_in_episode,
        'final_reward': args.pycolab_final_reward,
        'crop': args.pycolab_crop
    }
    env = batch_env.BatchEnv(args.num_processes, env_builder, **env_kwargs)

    ep_length = env.episode_length
    args.num_steps = ep_length
    envs = env
    obsspace = (3,5,5) #env.observation_shape
    actor_critic = Policy(
        obsspace,
        env.num_actions,
        base_kwargs={'recurrent': args.recurrent_policy})  # envs.observation_space.shape,
    actor_critic.to(device)

    ###################################
    ##decide on which underlying RL agent to use - ppo or a2c. But any other RL agent of choice should also work

    #if args.algo == 'ppo':
    agent = PPO(
        actor_critic,
        args.clip_param,
        args.ppo_epoch,
        args.num_mini_batch,
        args.value_loss_coef,
        args.entropy_coef,
        lr=args.lr,
        eps=args.eps,
        max_grad_norm=args.max_grad_norm)

    ###############CONSPEC FUNCTIONS##############################
    '''
    Here, the main ConSpec class is loaded. All the relevant ConSpec functions and objects are contained in this class.
    '''
    conspecfunction = ConSpec(args,   obsspace,  env.num_actions,  device)
    # file_path='datasets/frozen/trajectory_data_gfn_{}_seed_{}_{}_episodes_frozen.csv'.format(args.pycolab_game, args.seed, args.num_episodes)
    # cos_file_path='datasets/frozen/cos_sim_{}_seed_{}_{}_episodes_frozen.csv'.format(args.pycolab_game, args.seed ,args.num_episodes)

    ##############################################################
    print('steps', args.num_steps)
    rollouts = ReplayBuffer(args.num_steps, args.num_processes,
                              obsspace, env.num_actions,
                              actor_critic.recurrent_hidden_state_size, args.num_prototypes)  # envs.observation_space.shape
    rollouts.to(device)

    obs, _ = envs.reset()    
    obs = (torch.from_numpy(obs)).permute((0, 3, 1, 2)).to(device)
    rollouts.obs[0].copy_(obs)
    rollouts.to(device)

    episode_rewards = deque(maxlen=10)
    num_updates = int(args.num_env_steps) // args.num_steps // args.num_processes
    ##retrieving memories from the success/failure buffers
    all_memories = []
    logger.start()
    for episode in range(args.num_episodes):
        logger.step()
        obs, _ = envs.reset()
        obs = (torch.from_numpy(obs)).permute((0, 3, 1, 2)).to(device)
        rollouts.obs[0].copy_(obs)
        donetotal = np.zeros((args.num_processes,))  # .to(device)
        if args.use_linear_lr_decay:
            # decrease learning rate linearly
            update_linear_schedule(
                agent.optimizer, episode, num_updates,
                agent.optimizer.lr if args.algo == "acktr" else args.lr)

        for step in range(args.num_steps):
            # Sample actions
            with torch.no_grad():
                value, action, action_log_prob, recurrent_hidden_states = actor_critic.act(
                    rollouts.obs[step], rollouts.recurrent_hidden_states[step],
                    rollouts.masks[step])

            obs, reward = envs.step(action)
            obs = torch.from_numpy(obs).permute((0, 3, 1, 2)).to(device)
            reward = torch.from_numpy(reward).reshape(-1, 1)
            done = donetotal
            masks = torch.FloatTensor(
                [[0.0] if done_ else [1.0] for done_ in done])
            bad_masks = masks
            rollouts.insert(obs, recurrent_hidden_states, action,
                            action_log_prob, value, reward, masks, bad_masks)

        with torch.no_grad():
            next_value = actor_critic.get_value(
                rollouts.obs[-1], rollouts.recurrent_hidden_states[-1],
                rollouts.masks[-1]).detach()
            # now compute new rewards
            rewardstotal = rollouts.retrieveR()
            episode_rewards.append(rewardstotal.sum(0).mean().cpu().detach().numpy())

        ###############CONSPEC FUNCTIONS##############################
        '''
        the purpose here is to: 
        1. retrieve the current minibatch of trajectory (including its observations, rewards, hidden states, actions, masks)
        2. "do everything" that ConSpec needs to do internally for training, and output the intrinsic + extrinsic reward for the current minibatch of trajectories
        3. store this total reward in the memory buffer 
        '''
        
        obstotal, rewardtotal, recurrent_hidden_statestotal, actiontotal,  maskstotal  = rollouts.release()
        reward_intrinsic_extrinsic  = conspecfunction.do_everything(obstotal, recurrent_hidden_statestotal, actiontotal, rewardtotal, maskstotal)
        rollouts.storereward(reward_intrinsic_extrinsic)
        ##############################################################

        rollouts.compute_returns(next_value, args.use_gae, args.gamma,
                                 args.gae_lambda, args.use_proper_time_limits)

        value_loss, action_loss, dist_entropy= agent.update(rollouts)
        rollouts.after_update()

       
        if episode % args.log_interval == 0 and len(episode_rewards) > 1:
            logger.meter("results", "R", rewardstotal[-10:,:].sum(0).mean().cpu().detach().numpy())
            logger.meter("results", "dist_entropy", dist_entropy)
            logger.meter("results", "value_loss", value_loss)
            logger.meter("results", "action_loss", action_loss)
        
        if episode % args.checkpoint_interval == 0:
            buffer = {
            'obs': rollouts.obs,
             'rewards': rollouts.rewards,
             'hidden_states': rollouts.recurrent_hidden_states,
             'actions': rollouts.actions,
             'masks': rollouts.masks,
             'bad_masks': rollouts.bad_masks,
             'value_preds': rollouts.value_preds,
            }
            sf_buffer = conspecfunction.rollouts.retrieve_SFbuffer()
            conspec_rollouts = {
                'obs': sf_buffer[0],
                'rewards': sf_buffer[5],
                'hidden_states': sf_buffer[1],
                'actions': sf_buffer[3],
                'masks': sf_buffer[2],
                'bad_masks': sf_buffer[2],
                'value_preds': sf_buffer[4],
            }
            tensor_proto_list = [p.data for p in conspecfunction.prototypes.prototypes]
            checkpoint = {
                'epoch': episode,
                'encoder_state_dict': conspecfunction.encoder.state_dict(),
                'agent_state_dict': agent.actor_critic.state_dict(),
                'optimizer_conspec_state_dict': conspecfunction.optimizerConSpec.state_dict(),
                'optimizer_ppo_state_dict': agent.optimizer.state_dict(),
                'prototypes_state_dict': tensor_proto_list,
                }
            cos_checkpoint = {
                'cos_max_scores' : conspecfunction.rollouts.cos_max_scores, 
                'max_indices' : conspecfunction.rollouts.max_indx,
                'cos_scores' : conspecfunction.rollouts.cos_scores,
                # 'cos_success' : conspecfunction.rollouts.cos_score_pos,
                # 'cos_failure' : conspecfunction.rollouts.cos_score_neg,
            }

            checkpoint_path = os.path.join(full_path, f'checkpoint_epoch_{episode}.pth')
            buffer_path = os.path.join(full_path, f'buffer_epoch_{episode}.pth')
            conspec_rollouts_path = os.path.join(full_path, f'conspec_rollouts_epoch_{episode}.pth')
            cos_path = os.path.join(full_path, f'cos_sim_epoch_{episode}.pth')

            checkpoint_path = os.path.join(full_path, f'checkpoint_epoch_{episode}.pth')
            buffer_path = os.path.join(full_path, f'buffer_epoch_{episode}.pth')
            
            torch.save(checkpoint, checkpoint_path)
            print('checkpoint saved')
            torch.save(buffer, buffer_path)
            print('buffer saved')
            torch.save(conspec_rollouts, conspec_rollouts_path)
            print('success/failure buffers saved')
            torch.save(cos_checkpoint, cos_path)
            print('cosine similarity saved')
           
    logger.finish()

    # wandb.log()
    # wandb.finish()



if __name__ == "__main__":
    import argparse 
    from pathlib import Path
    parser = argparse.ArgumentParser(description='multi-key-door-conspec')
    parser.add_argument('--num_prototypes', type=int, default=8, help='number of prototypes(default: 8)')
    parser.add_argument('--choiceCLparams', type=int, default=0, help='CL params (default: 0)')
    parser.add_argument('--expansion', type=int, default=24, help='gail batch size (default: 24)')
    
    parser.add_argument('--pycolab_game', default='key_to_doormany4', help='name of the environment (default: key_to_doormany4)')
    parser.add_argument('--pycolab_apple_reward_min', type=float, default=0., help='pycolab apple reward min (default: 1.)')
    parser.add_argument('--pycolab_apple_reward_max', type=float, default=0., help='pycolab apple reward max (default: 2.)')
    parser.add_argument('--pycolab_final_reward', type=float, default=1., help='RMSprop optimizer epsilon (default: 1e-5)')
    parser.add_argument('--pycolab_num_apples', type=int, default=10, help='gail batch size (default: 128)')
    parser.add_argument('--pycolab_fix_apple_reward_in_episode', action='store_false', default=True, help='use generalized advantage estimation')
    parser.add_argument('--pycolab_crop', action='store_true', default=False, help='use generalized advantage estimation')
    
    parser.add_argument('--skip', type=int, default=4, help='gail batch size (default: 128)')
    #parser.add_argument('--lrCL', type=float, default=7e-4, help='Conspec learning rate (default: 7e-4)')
    parser.add_argument('--lrConSpec', type=float, default=2e-3, help='learning rate (default: 7e-4)')
    parser.add_argument('--algo', default='ppoCL', help='algorithm to use: a2c | ppo | acktr')
    parser.add_argument('--gail', action='store_true', default=False, help='do imitation learning with gail')
    parser.add_argument('--gail-experts-dir', default='./gail_experts', help='directory that contains expert demonstrations for gail')
    parser.add_argument('--gail-batch-size', type=int, default=128, help='gail batch size (default: 128)')
    parser.add_argument('--gail-epoch', type=int, default=5, help='gail epochs (default: 5)')
    parser.add_argument('--lr', type=float, default=7e-4, help='learning rate (default: 7e-4)')
    parser.add_argument('--eps', type=float, default=1e-5, help='RMSprop optimizer epsilon (default: 1e-5)')
    parser.add_argument('--alpha', type=float, default=0.99, help='RMSprop optimizer apha (default: 0.99)')
    parser.add_argument('--factorR', type=float, default=0.2, help='discount factor for rewards (default: 0.99)')
    parser.add_argument('--factorC', type=float, default=5000., help='discount factor for rewards (default: 0.99)')
    parser.add_argument('--gamma', type=float, default=0.99, help='discount factor for rewards (default: 0.99)')
    parser.add_argument('--use-gae', action='store_true', default=False, help='use generalized advantage estimation')
    parser.add_argument('--gae-lambda', type=float, default=0.95, help='gae lambda parameter (default: 0.95)')
    parser.add_argument('--entropy-coef', type=float, default=0.01, help='entropy term coefficient (default: 0.01)')
    parser.add_argument('--value-loss-coef', type=float, default=0.5, help='value loss coefficient (default: 0.5)')
    parser.add_argument('--max-grad-norm', type=float, default=0.5, help='max norm of gradients (default: 0.5)')
    parser.add_argument('--intrinsicR_scale', type=float, default=0.2, help='')
    
    parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')
    parser.add_argument('--cuda-deterministic', action='store_true', default=False, help="sets flags for determinism when using CUDA (potentially slow!)")
    parser.add_argument('--num_iterations', type=int, default=10000, help='Number of iterations (default: %(default)s)')
    parser.add_argument('--num-processes', type=int, default=16, help='how many training CPU processes to use (default: 16)')
    parser.add_argument('--num-steps', type=int, default=5, help='number of forward steps in A2C (default: 5)')
    parser.add_argument('--ppo-epoch', type=int, default=4, help='number of ppo epochs (default: 4)')
    parser.add_argument('--num-mini-batch', type=int, default=32, help='number of batches for ppo (default: 32)')
    parser.add_argument('--clip-param', type=float, default=0.2, help='ppo clip parameter (default: 0.2)')
  
    parser.add_argument('--num-env-steps', type=int, default=10e8, help='number of environment steps to train (default: 10e6)')
    parser.add_argument('--num_episodes', type=int, default=500, help='500 for 2 key, 1000 for 3 keys, 2000 for 4 keys')
    parser.add_argument('--env_name', default='PongNoFrameskip-v4', help='environment to train on (default: PongNoFrameskip-v4)')
    parser.add_argument('--no-cuda',action='store_true', default=False, help='disables CUDA training')
    parser.add_argument('--use-proper-time-limits', action='store_true', default=False, help='compute returns taking into account time limits')
    parser.add_argument('--recurrent-policy', action='store_false', default=True, help='use a recurrent policy')
    parser.add_argument('--use-linear-lr-decay', action='store_true', default=False, help='use a linear schedule on the learning rate')
    
    parser.add_argument('--log-interval', type=int, default=10, help='log interval, one log per n updates (default: 10)')
    parser.add_argument('--save-interval', type=int, default=100, help='save interval, one save per n updates (default: 100)')
    parser.add_argument('--eval-interval', type=int, default=None, help='eval interval, one eval per n updates (default: None)')
    parser.add_argument('--checkpoint_interval', type=int, default=500, help='checkpoint interval (default: 500)')
    parser.add_argument('--log-dir', default='/data/agent/', help='directory to save agent logs (default: /tmp/gym)')
    parser.add_argument('--save-dir', default='data', help='directory to save agent logs (default: ./trained_models/)')
    parser.add_argument('--output_folder', type=Path, default='output', help='Output folder (default: %(default)s)')
    parser.add_argument("--wandb_project",type=str,default='gfn-conspec',help="Wandb project name")
    parser.add_argument("--wandb_group",type=str,default='blake-richards',help="Wandb group name")
    parser.add_argument("--wandb_dir",type=str,default=f'{os.environ["SCRATCH"]}/exploringConsPec',help="Wandb logdir")
    parser.add_argument("--exp_name",type=str,default='exp-conspec',help="experiment name")
    parser.add_argument("--exp_group",type=str,default='exp-conspec',help="conspec")
    parser.add_argument("--exp_job_type",type=str,default='exp-conspec',help='conspec training')


    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    main(args)
    
