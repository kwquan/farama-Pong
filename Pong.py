import gymnasium as gym
import numpy as np
import collections
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from gymnasium.wrappers import GrayScaleObservation, FrameStack, ResizeObservation, TransformObservation
from tqdm import tqdm
from torch import optim

episodes = 800
batch_size = 32
start_epsilon = 1
min_epsilon = 0.1
epsilon_decay = 0.99999
gamma = 0.99 
learning_rate = 0.0001
min_len = 10000
max_len = 100000
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class DQN(nn.Module):
    def __init__(self):
        super().__init__()
        self.Conv1 = nn.Conv2d(4,4*8,8,stride=4,device=device)
        self.Conv2 = nn.Conv2d(4*8,4*8*2,4,stride=2,device=device)
        self.Conv3 = nn.Conv2d(64,64,3,stride=1,device=device)
        self.Linear1 = nn.Linear(3136,512)   
        self.Linear2 = nn.Linear(512,6)    
        
    def forward(self,x):
        x = F.relu(self.Conv1(x))
        x = F.relu(self.Conv2(x))
        x = F.relu(self.Conv3(x))
        x = torch.flatten(x,1,3)
        x = F.relu(self.Linear1(x))
        x = self.Linear2(x)
        return x

class ExperienceReplay:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)
        
    def __len__(self):
        return len(self.buffer)
    
    def append(self, experience):
        self.buffer.append(experience)
  
    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer),batch_size,replace=False)
        states, actions, rewards, dones, next_states = zip(*[self.buffer[idx] for idx in indices])
        return np.array(states), np.array(actions,dtype=np.int64), np.array(rewards,dtype=np.float32), np.array(dones, dtype=np.uint8), np.array(next_states)
    
class Agent:
    def __init__(self, mode="training"):
        self.env = env
        self.epsilon = start_epsilon
        self.device = device
        self.buffer = buffer
        self.model = q_net
        self.target_model = q_target
        self.mode = mode
        self.episode = 0
        self.learns = 0
        self._reset()

    def _reset(self):
        self.state, _ = self.env.reset(seed=0)
        self.timestep = 0
        self.total_reward = 0      

    def select_action(self):
        if self.mode == "training":
            if np.random.random() < self.epsilon:
                action = self.env.action_space.sample()
            else:
                state = np.array([self.state])
                state = torch.tensor(state).to(self.device)
                action = np.argmax(self.model(state).cpu().detach().numpy())
        else:
            state = np.array([self.state])
            state = torch.tensor(state).to(self.device)
            action = np.argmax(self.model(state).cpu().detach().numpy())       
        return action      
    
    def get_experience(self):
        episode_reward = None
        action = self.select_action()
        next_state, reward, terminate, _, _ = self.env.step(action)
        exp = Experience(self.state,action,reward,terminate,next_state)
        self.buffer.append(exp)
        self.state = next_state
        self.timestep += 1
        self.total_reward += reward
        
        if terminate:
            episode_reward = self.total_reward
            print(f"timestep {self.timestep} Score: {episode_reward}")
            self.episode += 1
            self._reset()
            return True, episode_reward 
        
        if len(agent.buffer) >= min_len:
            self.update_weights() 

        return False, episode_reward 
    
    def update_weights(self):
        batch = self.buffer.sample(batch_size)
        states, actions, rewards, dones, next_states = batch
                
        states_t = torch.tensor(states).to(self.device)
        next_states_t = torch.tensor(next_states).to(self.device)
        actions_t = torch.tensor(actions).to(self.device)
        rewards_t = torch.tensor(rewards).to(self.device)
        done_mask = torch.BoolTensor(dones).to(self.device)
        action_values = self.model(states_t).gather(1,actions_t.unsqueeze(-1)).squeeze(-1)
        next_action_values = self.target_model(next_states_t).max(1)[0]
        next_action_values[done_mask] = 0.0
        next_action_values = next_action_values.detach()
        
        expected_action_values = rewards_t + next_action_values*gamma 
        loss_t = nn.MSELoss()(action_values, expected_action_values)
        
        optimizer.zero_grad()
        loss_t.backward()
        optimizer.step()
        self.learns += 1
            
        if self.learns % 1000 == 0:
            self.target_model.load_state_dict(self.model.state_dict())
            print(f"episode {self.episode}: target model weights updated")


env = FrameStack(TransformObservation(ResizeObservation(GrayScaleObservation(gym.make('ALE/Pong-v5', render_mode='human', full_action_space=False)), (110,84)), lambda x: np.array(x[18:102,:]).astype(np.float32) / 255.0  ), 4)
Experience = collections.namedtuple('Experience',field_names=['state', 'action', 'reward', 'done', 'next_state'])
buffer = ExperienceReplay(min_len)
q_net = DQN().to(device)
q_target = copy.deepcopy(q_net).to(device)
epsilon = start_epsilon
episode_rewards = []
agent = Agent()
optimizer = optim.Adam(agent.model.parameters(), lr=learning_rate)

for episode in tqdm(range(episodes)):
    terminate = False
    while not terminate:
        agent.epsilon = max(agent.epsilon*epsilon_decay,min_epsilon)
        terminate, reward = agent.get_experience()
        if terminate:
            episode_rewards.append(reward)
            mean_reward = round(np.mean(episode_rewards[-100:]),3)
            print(f"episode {episode}, mean reward: {mean_reward}")

env.reset()        
env.close() 

torch.save(agent.model.state_dict(), "pong_agent.pth")
fig, ax = plt.subplots()
ax.plot(np.arange(len(episode_rewards)), episode_rewards)
ax.set(xlabel='timestep', ylabel='episode reward',
    title="reward over time")
ax.grid()
fig.savefig("pong_result.png")

