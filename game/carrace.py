import numpy as np
import gym

from game.agent.agent import Agent, TDAgent
from game.agent.encoder import CarRaceEncoder
from game.base import Game

if __name__ == '__main__':

  game = Game(
    gymenv="CarRacing-v0",
    path="model/carrace-td.agent",
    init_model=TDAgent(encoder=CarRaceEncoder(), learning_rate=0.8, alpha=0.9),
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
    train_kmeans_after=10,
    train_kmeans_every=3)

  game.run(n_episodes=50000)