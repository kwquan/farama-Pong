# Pong
This repo contains the code used to solve the Pong[V5] environment using Deep Q-learning. 
based on this paper(https://arxiv.org/abs/1312.5602) with a few tweaks(due to hardware limitations):
1) Simplified Q network architecture
2) Used epsilon decay instead of linear decreasing epsilon over 1 million episodes
3) Used Adam optimizer instead of RMSprop
4) Trained for only 800 episodes
5) Min length of buffer set to 10000
6) Normalized pixel values before passing to Q net

# DQN
### get experience
![alt text](https://github.com/kwquan/farama-Pong/blob/main/get_experience.png)

### update weights
![alt text](https://github.com/kwquan/farama-Pong/blob/main/update_weights.png)

# Results
![alt text](https://github.com/kwquan/farama-Pong/blob/main/pong_result.png)

As observed, mean reward(for latest 100 episodes) gradually increases. With better tuning & more episodes, results can be improved
