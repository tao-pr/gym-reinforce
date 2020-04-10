import numpy as np
import gym

from game.agent.agent import Agent, TDAgent
from game.agent.encoder import CarRaceEncoder
from game.base import Game

if __name__ == '__main__':

  # Tips for Car racing (with Temporal-Difference)
  # - Keep learning rate low    => So good policy needs to repeat to confirm, also the mistakes
  # - Keep alpha high           => So future has high influence
  # - Break early               => Otherwise we learn too much about decreasing neighbour states to grass
  # - Update KMeans often       => This helps picking up good improvisation


  game = Game(
    gymenv="CarRacing-v0",
    path="model/carrace-td.agent",
    init_model=TDAgent(
      encoder=CarRaceEncoder(),
      learning_rate=0.8, alpha=0.9,
      num_state_clusters=8),
    actions=[np.array(v) for v in [
      # [steer, gas, brake]
      [-0.5, 0.1, 0],
      [ 0.5, 0.1, 0],
      [-0.2, 0.1, 0],
      [ 0.2, 0.1, 0],
      [-0.2, 0.3, 0],
      [ 0.2, 0.3, 0],
      [   0, 0.5, 0],
      [   0, 0.1, 0]
    ]],
    protect_first_iters=100,
    max_consecutive_decrease=5,
    train_kmeans_after=10,
    train_kmeans_every=3)

  game.run(n_episodes=50000)