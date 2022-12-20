import sys, os
sys.path.insert(0, os.path.abspath(".."))
os.environ["MUJOCO_GL"] = "egl" # for mujoco rendering
import time
from pathlib import Path

import torch
import gym
import hydra
import wandb
import warnings
warnings.filterwarnings("ignore", category=UserWarning) 
warnings.filterwarnings("ignore", category=DeprecationWarning) 

from ddpg import DDPG
#from pg_ac import PG
from common import helper as h
from common import logger as logger
from dqn import DQNAgent
from ddqn import DDQNAgent
from common.buffer import ReplayBuffer
import numpy as np
import yaml


GYM_TASKS = {
    'MountainCarContinuous-v0',
    'LunarLander-v2',
    'BipedalWalker-v3',
    'Hopper-v2',
    'Hopper-v4'
}


# Creates the environment for LunarLander and MountainCar continous

def create_env(config_file_name, seed):
    config = yaml.load(open(f'./configs/{config_file_name}.yaml', 'r'),  Loader=yaml.Loader)

    if config['env_name'] in GYM_TASKS:
        env_kwargs = config['env_parameters']
        if env_kwargs is None:
            env_kwargs = dict()
        env = gym.make(config['env_name'], **env_kwargs)
        env.reset(seed=seed)

    else:
        raise NameError("Wrong environment name was provided in the config file! Please report this to @Nikita Kostin.")

    return env

def to_numpy(tensor):
    return tensor.cpu().numpy().flatten()



class ContinuousToDiscrete:
    def __init__(self, n_bins, low, high):
        self.values = np.linspace(low, high, n_bins)

    def __call__(self, continuous_input):
        return int(round((continuous_input + 1) / 2 * self.values.shape[0] + 0))

    def back(self, discrete_input):
        return self.values[discrete_input]


class DiscretizedActionWrapper(gym.Wrapper):
    def __init__(self, env, n_bins):
        super(DiscretizedActionWrapper, self).__init__(env)
        # self.trafo = discretize(env.action_space, n_bins)
        # self.action_space = self.trafo.target
        # self.action = self.trafo.convert_from
        self.transform = ContinuousToDiscrete(
            low=self.action_space.low[0], high=self.action_space.high[0], n_bins=n_bins)
        self.action_space = gym.spaces.Discrete(n_bins)

    def step(self, action):
        action = np.array([self.transform.back(action)])
        next_state, reward, done, info = super(DiscretizedActionWrapper, self).step(action)
        next_state = next_state.flatten()

        return next_state, reward, done, info


# Creates environment for MountainCar discrete given by Nikita
def create_env_mountain_car_discrete(config_file_name, seed, n_bins=10): #10
    config = yaml.load(open(f'./configs/{config_file_name}.yaml', 'r'),  Loader=yaml.Loader)

    if config['env_name'] in GYM_TASKS:
        env_kwargs = config['env_parameters']
        if env_kwargs is None:
            env_kwargs = dict()
        env = gym.make(config['env_name'], **env_kwargs)
        if config_file_name.split('_')[0] == 'mountaincardiscrete':
            env = DiscretizedActionWrapper(env, n_bins=n_bins)
        env.reset(seed=seed)

    else:
        raise NameError("Wrong environment name was provided in the config file! Please report this to @Nikita Kostin.")

    return env



# The main function
@hydra.main(config_path='cfg', config_name='cfg_a')
def main(cfg):
    # sed seed
    h.set_seed(cfg.seed)
    cfg.run_id = int(time.time())

    # create folders if needed
    work_dir = Path().cwd()/'results'/f'{cfg.env_name}'
    if cfg.save_model: h.make_dir(work_dir/"model")
    if cfg.save_logging: 
        h.make_dir(work_dir/"logging")
        L = logger.Logger() # create a simple logger to record stats

    # Model filename
    if cfg.model_path == 'default':
        cfg.model_path = work_dir/'model'/f'{cfg.env_name}_{cfg.agent_name}_{cfg.seed}_weights.pt'

    # use wandb to store stats; we aren't currently logging anything into wandb during testing
    if cfg.use_wandb and not cfg.testing:
        wandb.init(project="rl_aalto",
                    name=f'{cfg.exp_name}-{cfg.env_name}-{str(cfg.seed)}-{str(cfg.run_id)}',
                    group=f'{cfg.exp_name}-{cfg.env_name}',
                    config=cfg)


    #  Create LunarLander medium environment
    if cfg.env_name == 'LunarLander' and  cfg.agent_name == 'ddpg':
        env = create_env(config_file_name = 'lunarlander_continuos_medium', seed=cfg.seed) 
    elif cfg.env_name=='LunarLander' and (cfg.agent_name=='dqn' or cfg.agent_name=='ddqn'):
        env = create_env(config_file_name = 'lunarlander_discrete_medium', seed=cfg.seed)


    # Create MountainCar easy environment
    if cfg.env_name=='MountainCar' and cfg.agent_name=='ddpg':
        env = create_env(config_file_name = 'mountaincarcontinuous_easy', seed = cfg.seed) 
    if cfg.env_name=='MountainCar' and (cfg.agent_name=='dqn' or cfg.agent_name=='ddqn'):
        env = create_env_mountain_car_discrete(config_file_name = 'mountaincardiscrete_easy', seed=cfg.seed)


    if cfg.save_video:
        # During testing, save every episode
        if cfg.testing:
            ep_trigger = 1
            video_path = work_dir/'video'/'test'
        # During training, save every 50th episode
        else:
            ep_trigger = 50
            video_path = work_dir/'video'/'train'
        env = gym.wrappers.RecordVideo(env, video_path,
                                        episode_trigger=lambda x: x % ep_trigger == 0,
                                        name_prefix=cfg.exp_name) # save video every 50 episode


    
    # Create LunarLander and dqn agent

    if cfg.agent_name == "dqn" and cfg.env_name=='LunarLander':
        n_actions = env.action_space.n
        state_shape = env.observation_space.shape

        agent = DQNAgent(state_shape, n_actions, batch_size=512, hidden_dims=[64,64],
                         gamma=cfg.gamma, lr=5e-5, tau=0.001)
        buffer = ReplayBuffer(state_shape, action_dim=1, max_size=int(1e6))

    # Create LunarLnader and ddqn agent 

    elif cfg.agent_name == "ddqn" and cfg.env_name=='LunarLander':
        n_actions = env.action_space.n
        state_shape = env.observation_space.shape

        agent = DDQNAgent(state_shape, n_actions, batch_size=512, hidden_dims=[64,64],
                         gamma=cfg.gamma, lr=5e-5, tau=0.001)
        buffer = ReplayBuffer(state_shape, action_dim=1, max_size=int(1e6))

    # Create LunarlLander and ddpg agent 
    elif cfg.agent_name == "ddpg" and cfg.env_name=='LunarLander':
        state_shape = env.observation_space.shape
        action_dim = env.action_space.shape[0]
        max_action = env.action_space.high[0]
        agent = DDPG(state_shape, action_dim,max_action, lr = 3e-4, gamma=0.99, tau=0.005, batch_size=400, buffer_size=1e6)
    

    # Create MountainCar and dqn 

    if cfg.agent_name=="dqn" and cfg.env_name=='MountainCar':
        n_actions = env.action_space.n
        state_shape = env.observation_space.shape

        agent = DQNAgent(state_shape, n_actions, batch_size=256, hidden_dims=[1024,1024,1024],
                         gamma=cfg.gamma, lr=0.001, tau=0.001)
        buffer = ReplayBuffer(state_shape, action_dim=1, max_size=int(2e6)) 

    # Create MountainCar and ddqn
    
    elif cfg.agent_name=='ddqn' and  cfg.env_name=='MountainCar':
        n_actions = env.action_space.n
        state_shape = env.observation_space.shape

        agent = DDQNAgent(state_shape, n_actions, batch_size=256, hidden_dims=[1024,1024,1024],
                         gamma=cfg.gamma, lr=0.001, tau=0.001)
        buffer = ReplayBuffer(state_shape, action_dim=1, max_size=int(2e6))

    
    # Create MountainCar ddpg

    elif cfg.agent_name=='ddpg' and cfg.env_name=='MountainCar':
        state_shape = env.observation_space.shape
        action_dim = env.action_space.shape[0]
        max_action = env.action_space.high[0]
        agent = DDPG(state_shape, action_dim,max_action, lr = 0.001, gamma=0.98, tau=0.005, batch_size=128, buffer_size=2e6, \
                    agent='MountainCar')
    

    




    if not cfg.testing: # training
        for ep in range(cfg.train_episodes + 1):
            # collect data and update the policy
            if cfg.agent_name=='dqn' or cfg.agent_name=='ddqn':
                train_info = train_dqn(agent, env, ep, cfg, buffer) 
            else:
                train_info = train(agent, env)

            if cfg.use_wandb:
                wandb.log(train_info)
            if cfg.save_logging:
                L.log(**train_info)
            if (not cfg.silent) and (ep % 100 == 0):
                print({"ep": ep, **train_info})

        
        if cfg.save_model:
            agent.save(cfg.model_path)

    else: # testing
        if cfg.model_path == 'default':
            cfg.model_path = work_dir/'model'/f'{cfg.env_name}_{cfg.agent_name}_{cfg.seed}_weights.pt'
        print("Loading model from", cfg.model_path, "...")

        # load model
        agent.load(cfg.model_path)
        
        print('Testing ...')
        if cfg.agent_name=='ddpg':
            test(agent, env, num_episode=50)
        else: 
            test_dqn(agent, env, num_episode=50)


# Trains environment usin DDPG

def train(agent, env, max_episode_steps=1000):
    # Run actual training        
    reward_sum, timesteps, done, episode_timesteps = 0, 0, False, 0
    # Reset the environment and observe the initial state
    obs = env.reset()
    while not done:
        episode_timesteps += 1
        
        # Sample action from policy
        action, act_logprob = agent.get_action(obs)

        # Perform the action on the environment, get new state and reward
        next_obs, reward, done, _ = env.step(to_numpy(action))

#         # Store action's outcome (so that the agent can improve its policy)
#         if isinstance(agent, PG):
#             done_bool = done
#             agent.record(obs, act_logprob, reward, done_bool, next_obs)

        if isinstance(agent, DDPG):
            # ignore the time truncated terminal signal
            done_bool = float(done) if episode_timesteps < max_episode_steps else 0 
            agent.record(obs, action, next_obs, reward, done_bool)
        else: raise ValueError

        # Store total episode reward
        reward_sum += reward
        timesteps += 1

        # update observation
        obs = next_obs.copy()

    # update the policy after one episode
    info = agent.update()

    # Return stats of training
    info.update({'timesteps': timesteps,
                'ep_reward': reward_sum,})
    return info


# Testest environment using DDPG 
@torch.no_grad()
def test(agent, env, num_episode=10):
    total_test_reward = 0
    test_rewards = []
    for ep in range(num_episode):
        obs, done= env.reset(), False
        test_reward = 0
        
        while not done:
            action, _ = agent.get_action(obs, evaluation=True)
            obs, reward, done, info = env.step(to_numpy(action))
            
            test_reward += reward
            

        total_test_reward += test_reward
        test_rewards.append(test_reward)
        print("Test ep_reward:", test_reward)
    print("Average test reward:", total_test_reward/num_episode)
    print("SD: ", np.std(np.array(test_rewards)))



# Trains environments using dqn and ddqn 

def train_dqn(agent, env, ep, cfg, buffer):
    state, done, ep_reward, env_step = env.reset(), False, 0, 0
    eps = max(cfg.glie_b/(cfg.glie_b + ep), 0.05)

    # collecting data and fed into replay buffer
    while not done:
        env_step += 1
        if ep < cfg.random_episodes: # in the first #random_episodes, collect random trajectories
            action = env.action_space.sample()
        else:
            # Select and perform an action
            action = agent.get_action(state, eps)
            if isinstance(action, np.ndarray): action = action.item()
            
        next_state, reward, done, _ = env.step(action)
        ep_reward += reward

        # Store the transition in replay buffer
        buffer.add(state, action, next_state, reward, done)

        # Move to the next state
        state = next_state
    
        # Perform one update_per_episode step of the optimization
        if ep >= cfg.random_episodes:
            update_info = agent.update(buffer)
        else: update_info = {}

    info = {'episode': ep, 'epsilon': eps, 'ep_reward': ep_reward}
    info.update(update_info)
    return info



# Tests environment using dqn and ddqn 

@torch.no_grad()
def test_dqn(agent, env, num_episode=50):
    total_test_reward = 0
    test_rewards = []
    for ep in range(num_episode):
        state, done, ep_reward, env_step = env.reset(), False, 0, 0
        rewards = []

        # collecting data and fed into replay buffer
        while not done:
            # Select and perform an action
            action = agent.get_action(state, epsilon=0.0)
            if isinstance(action, np.ndarray): action = action.item()
            state, reward, done, _ = env.step(action)
            ep_reward += reward
            rewards.append(reward)

        info = {'episode': ep, 'ep_reward': ep_reward}
        print(info)
        test_rewards.append(ep_reward)
        total_test_reward += ep_reward
    print("Average test reward:", total_test_reward/num_episode)
    print("SD: ", np.std(np.array(test_rewards)))

# Entry point of the script
if __name__ == "__main__":
    main()
