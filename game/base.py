import numpy as np
import gym

from game.agent.agent import Agent, TDAgent
from game.agent.encoder import *

class Game:

  def __init__(self, 
    gymenv="CarRacing-v0",
    path="dummy.agent",
    init_model=TDAgent(encoder=CarRaceEncoder(), learning_rate=0.8, alpha=0.9),
    actions=[],
    train_kmeans_after=10,
    train_kmeans_every=3):

    self.env = gym.make(gymenv)
    self.agent = Agent.load(path, init_model)
    self.actions = actions

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

        if done or ((total_reward <= 0 or num_consecutive_reduction > 5) and n > 300):
          print("... Episode DONE!")
          print("... The agent knows {} observations so far".format(len(self.agent.v)))
          self.agent.encoder.n = 0
          done = True

          if i%self.train_kmeans_every==0 and i>self.train_kmeans_after:
            # Rebuild K-Means clusters every N episodes
            self.agent.revise_clusters()

          # Save the trained agent
          self.agent.save(self.path)

      print("Best score so far : ", best_reward)

  
