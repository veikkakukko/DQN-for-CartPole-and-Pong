# DQN-for-CartPole-and-Pong
Reinforcement learning: CartPole and Pong with a DQN network

This is a coursework on the Artificial Intelligence course at the School of Computing at UEF. This document contains the report of the project.

##Problem description:##

The problem is to get an AI agent to learn to play games like CartPole and Pong with
reinforcement learning. The idea is to make the agent learn the best actions in every state based
on rewards (the game score). It is very challenging to make the agent to learn, because the high-
dimensional input (raw pixels in Pong) is heavy to handle and there’s a long delay in rewards
(Pong: game score at the end of an episode). To reach good results in relatively short time one
has to use neural networks, but their instability can make training frustrating.


##Approach to Solution:##
This project work's assignment is based on the original Deep Q Learning paper by Google
DeepMind. The paper includes a pseudo code algorithm that is well explained in the paper. I
used that as a base to start building my own version of a DQN agent. The main idea of the
paper is to use a deep neural network to learn Q-values (expected rewards of actions taken in a
certain state). The algorithm uses a decaying epsilon value to determine the probability of
random actions for exploration. State-action pairs and state transitions are stored in a replay
memory. Random samples from the memory rewards of transitions are used to evaluate the
fitness of the agent. Loss is calculated and network weights optimized accordingly. Every now
and then the network weights are copied to a target network that can be used for comparison
and to stabilize learning.


##Implementation:##
The DQN paper, where the idea of reinforcement learning with a neural network was
introduced, seemed very complicated to me at first. I had to read it many times to understand
the basic algorithm. I also asked ChatGPT to explain me some details and I watched a lot of
tutorials from YouTube to grasp the concept.

The best option for this project was to use PyTorch, because it oﬀers nice flexibility and
versatility for this kind of project. The game simulations were performed in OpenAI’s gym
environment. I implemented a DQN first for CartPole and then for Pong. Pong-v4 wasn’t
available in Colab for unknown reasons, but I used ”ALE/Pong-v5” instead.

I had major diﬃculties with CartPole. I couldn’t get the model to learn at first. Everything
seemed to be in place, but the model had a flat learning curve. After tweaking the parameters
dozens of times and running the training again and again I finally compared my code to
PyTorch Reinforcement Learning Tutorial. I read both codes one line at a time to see that they
have the same basic functionalities. I realized that I was using an inverse of the target update
rate but in the training loop I was treating it as just the target update rate. So that parameter
was oﬀ by a factor of 40 000. I changed it and finally got to see a learning curve. I tuned the
parameters a little bit to make the agent learn CartPole in 500 episodes.

While building the version to train my agent in Pong, I had diﬃculties with the Colab runtime
service. I had two sessions running at the same time and I tried to use GPU for both. It wasn’t
possible and finally Colab denied my access to any GPUs at all. With a CPU the training
would’ve taken days. I tried again the next day but no GPUs were available. I had to buy a
Colab Pro subscription to be able to train my models.

I also learned a valuable lesson about saving the model. My estimation about the training of my
Pong-model was 5-6 hours. I added the line of code to save the model after the training was
complete. However, my model seemed to learn Pong in about 1000 steps, but the runtime got
cut oﬀ halfway through. The model wasn’t saved, because all session data was deleted when the
session was disconnected. After that I added an automated save to Google Drive every 100
training episodes.

Separate analysis with learning curves and hyper-parameters of CartPole and Pong can be found in the next parts. Structure of my project is presented after them.


##Analysis of learning CartPole##
I may have had an overfitting problem with CartPole. A few training sessions had a curve that reached
episode durations of 500 for a while (the point where the agent balances the pole perfectly), but then
came down to around one hundred or even below that. I tried changing the learning rate, epsilon decay
rate and target net update rate, but couldn’t achieve very stable learning. I tried using the Double DQN
method for steadier learning, but it ended up being worse. I kept to the soft update method that I got
from the PyTorch DQN tutorial. With it I achieved the result shown below. Still not very stable, but at
least the algorithm worked somehow.

<img width="1114" height="878" alt="image" src="https://github.com/user-attachments/assets/98d7f732-1d57-4e60-afd4-eec9f220ad6b" />

My hyper-parameters used for CartPole:
BATCH SIZE = 128
GAMMA = 0.99
EPSILON_START = 0.95 # DECAYING EPSILON 0.95 -> 0.05 WITH 2500 STEPS
EPSILON_END = 0.05
EPSILON_DECAY = 2500
NUM_EPISODES = 500
TAU = 0.005 # TARGET NETWORK GETS UPDATED EVERY 200 STEPS
MIN_REPLAY_SIZE = 1000


##Analysis of learning Pong:##
The training for Pong with 5000 episodes took 9 hours. The learning curve reached very good values in
a few hundred episodes and seemed to stay roughly on that level since. The reward from a single
episode is determined by the diﬀerence in the game score. Reaching 21 means that the agent played a
perfect game of 21-0. The moving average of 50 episodes stayed roughly between 16 and 18. From what I
read about training for Pong, my understanding is that an average reward of >16 means that the model
has mastered Pong. The learning curve plotted by rewards is shown below.

<img width="1182" height="936" alt="image" src="https://github.com/user-attachments/assets/8d9ecd32-b29c-4033-be61-6e2bfdd5c798" />

After training I performed ten episodes of evaluation with ε=0.0 greedy policy. My agent played ten
perfect games of 21-0, which means that it really had mastered Pong.
My hyper-parameters used for Pong:

BATCH_SIZE = 128
GAMMA = 0.99
EPSILON_START = 0.95 # DECAYING EPSILON 0.95 -> 0.05 WITH 100000 STEPS
EPSILON_END = 0.05
EPSILON_DECAY = 100000
NUM_EPISODES = 5000 # TOTAL EPISODES
TARGET_UPDATE = 1000 # TARGET NETWORK UPDATE RATE
MIN_REPLAY_SIZE = 1000


##Here is the basic structure of my project code:##

CLASSES:
  class DQN
  class ReplayBuﬀer

CONSTANTS/HYPERPARAMETERS:
  batch size,
  gamma,
  epsilon,
  number of episodes,
  target update rate,
  etc.

FUNCTIONS:
  def select_action
  def preprocess
  def plot_durations
  def plot_rewards
  def optimize_model

TRAINING LOOP:
  set up environment
  initialize replay memory
  set up Q-network
  set up target network
  define optimizer
    for episodes:
      get initial state
      repeat:
        select action based on state’s Q-value
        execute action in simulation
        observe
        update rewards
        store transition to replay memory
        move to next state
        optimize Q-network
        update target network by update rate
        if episode done, break


##Intelligence:##
Depending on the definition of intelligence, my agent is in some ways intelligent and in a lot of
ways not. The agent masters Pong with a few hours of training, but everything is based on
calculations and the model is essentially just a computer program. It seems like the agent is very
good at one thing, but it’s intelligence is still very narrow.

The learning curves shown previously show how the agent learned. The agent went from
completely random actions (I don’t have a recording of that) to playing perfect games of Pong,
so there is a big diﬀerence. A recording of one of the perfect games will be submitted as an
attachment to the report.


##Quotations:##

The DQN paper:
**https://www.datascienceassn.org/sites/default/files/Human-level%20Control
%20Through%20Deep%20Reinforcement%20Learning.pdf**

PyTorch DQN Tutorial:
**https://docs.pytorch.org/tutorials/intermediate/reinforcement_q_learning.html**
I used this tutorial for some parts of the algorithm that I couldn’t figure out myself. For
CartPole, I used soft update the same way this tutorial does. I took direction for a good set of
hyper-parameters from this tutorial and followed the plotting function.

ChatGPT:
I used ChatGPT for questions like: ”How to solve this error?”, ”How to convert numpy-arrays
to tensors?”, ”How to plot durations?” and ”What kind of behavior should I expect with these
hyper-parameters?”
ChatGPT helped me create a manual version of FrameStack when the original library wasn’t
available in Colab, and a trivial plotting function.

Gemini:
I used Gemini in Colab to help me format code and solve errors. In the end it caused more
harm than good, because it kept wanting to change other parts of the code when I was trying to
resolve one error at a time.

Contributions:
I did the whole project and the report alone. Even though there were a lot of diﬃculties, I’m
satisfied with the result.
After all the work using the DQN algorithm for Atari games feels relatively trivial. I’d imagine
similar techniques are used for much more complicated tasks with many more state parameters
to take into account. I am intrigued by the idea of using Deep Q-learning for more complicated
video game agents. I’d like to try that some day.
