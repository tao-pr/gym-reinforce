import numpy as np
import gym

from game.agent.agent import Agent, TDAgent
from game.agent.encoder import *

class Game:

  # Tips for Car racing (with Temporal-Difference)
  # - Keep learning rate high   => So good policy will be quickly learnt, and reinforced over time
  # - Keep alpha high           => So future has high influence
  # - Break early               => Otherwise we learn too much about decreasing neighbour states to grass
  # - Update KMeans often       => This helps picking up good improvisation

  def __init__(self, 
    gymenv="CarRacing-v0",
    path="dummy.agent",
    init_model=TDAgent(encoder=CarRaceEncoder(), learning_rate=0.8, alpha=0.9),
    actions=[],
    max_iter=1500,
    protect_first_iters=300,
    max_consecutive_decrease=7,
    train_kmeans_after=10,
    train_kmeans_every=2):

    self.env = gym.make(gymenv)
    self.agent = Agent.load(path, init_model)
    self.actions = actions
    self.max_iter = max_iter
    self.protect_first_iters = protect_first_iters
    self.max_consecutive_decrease = max_consecutive_decrease

    self.path = path
    self.train_kmeans_every = train_kmeans_every
    self.train_kmeans_after = train_kmeans_after

    print("Env created")
    print("Agent knows {} observations".format(len(self.agent.v)))

  def run(self,n_episodes=5000):

    # Start!
    print("Starting the learning episodes")
    best_reward = 0
    for i in range(n_episodes):
      
      observation = self.env.reset()
      print("Episode {} of {} ...".format(i+1, n_episodes))
      
      n = 0
      done = False
      last_action = -1
      last_state = None
      total_reward = 0

      last_reward = 0
      num_consecutive_reduction = 0

      while not done:
        n = n+1
        self.env.render()
        action,_ = self.agent.best_action(observation)

        # If the bot does not know how to react,
        # random from the action space
        if action == -1:
          # Take random action, blindly
          action = self.actions[np.random.choice(len(self.actions))]
        elif action is None:
          # Random action too
          action = self.env.action_space.sample()
        else:
          action = self.agent.encoder.decode_action(action)

        new_observation, reward, done, info = self.env.step(action)
        total_reward += reward

        if reward <= last_reward:
          num_consecutive_reduction += 1
        else:
          num_consecutive_reduction = 0

        last_reward = reward

        # Record best score
        if total_reward > best_reward:
          best_reward = total_reward

        # Learn
        self.agent.learn(observation, action, reward, new_observation)

        observation = new_observation

        if done or n > self.max_iter or ((total_reward < 0 or num_consecutive_reduction > self.max_consecutive_decrease) and n>self.protect_first_iters):
          print("... Episode DONE!")
          print("... The agent knows {} observations so far".format(len(self.agent.v)))
          self.agent.encoder.n = 0
          done = True

          if i%self.train_kmeans_every==0 and i>self.train_kmeans_after:
            # Rebuild K-Means clusters every N episodes
            self.agent.revise_clusters()

          # Save the trained agent
          self.agent.save(self.path)

          # Save the report, one line per episode
          with open("{}.log".format(self.path), "a") as f:
            f.write("{},{},{}\n".format(
              n, # TOL
              total_reward, # Total reward 
              len(self.agent.v) # Observation count
              ))

      print("Best score so far : ", best_reward)

  
