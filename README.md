![pong](https://i.ytimg.com/vi/ncB0ov5hT48/hqdefault.jpg)


# Reinforcement Learning Workshop
## Prerequisites

prior running the code, make sure to have the following installed:
```
torch==1.0.0
torchvision==0.2.1
tensorboard==1.12.2
tensorboardX==1.6
tensorflow==1.13.0rc2
opencv-python==4.0.0.21
gym==0.11.0
gym[atari]==0.11.0
numpy==1.15.4
```

to make sure everything is installed, just run
```
pip install -r requirements
```

## Assignments
Under `rl-workshop/` you will find 3 assignments:
1. DQN for pong (`/dqn`)
2. REINFORCE for cartpole (`/reinforce`)
3. Advantage Actor-Critic for pong (`/actor_critic`)

You are required to complete the missing code pieces 
(annotated by `...`, `YOUR CODE HERE`, or specific instructions).  
After completing the code, you can execute each assignment and debug 
it using tensorboard. Each execution will write scalars to a separate 
directory under `runs/`, and that will be the directory when executing 
`tensorboard --logdir DIR`.  
Since learning to play pong may take a long time, you should consider the task
as successful if you can see a steady raise of the rewards. Cartpole should
be solved in a matter of seconds.

## Rendering Pong
While training you pong agent using DQN, the best model is saved to `dqn/PongNoFrameskip-v4-best.dat`.
To visualize the game play using your best model, simply run 
```
python dqn/dqn_play_pong.py --model dqn/PongNoFrameskip-v4-best.dat
``` 

## Common code
There are some important utility classes under `lib/`:  
`experience.py:DiscountedExperienceSource` - given agent and environment,
will produce (s, a, r, s') tuples. It can work with a list of environments,
and with step size of more than 1 - This will produce a discounted reward for
the amount of steps taken.

`agents.py:PolicyAgent` - Implementation of an agent, that given a probability
distribution, will sample an action. **NOTE THAT YOU WILL HAVE TO IMPLEMENT PART
OF THIS CODE**

`wrappers.py` - common environment wrappers that are used for most atari environments.
You don't really need to get into much detail there.

`common.py` - helpful trackers that follow rewards, produce nice logging output,
and determine whether the game was solved.


We are here to answer any of your questions,  
Ilay and Jonathan