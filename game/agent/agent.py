import os
import numpy as np
import pandas as pd
import joblib
from sklearn.cluster import KMeans
from collections import Counter

from game.agent.encoder import *

class Agent:
  """
  Base reinforcement learning agent with basic interface
  """
  def __init__(self, learning_rate=0.25, num_state_clusters=8, actions=[]):
    self.learning_rate = learning_rate

    # Two-level dictionaries
    self.v = dict() # [state => reward]
    self.state_machine = dict() # [state => state' => action]

    # Binding state and action encoder (np.array => str)
    self.encoder = StateActionEncoder() # TODO Switch to its child class

    # Clusters of 
    self.num_state_clusters = num_state_clusters
    self.stateHashToState = dict()
    self.kmeans = None
    self.cluster_best_actions = dict()

    self.actions = actions


  def revise_clusters(self):
    """
    Given the knowledge of the observations so far,
    re-train the cluster of states and their most likelihood of best action
    """
    dataset = []
    best_action = []

    print("Building {} state clusters from {} states".format(
      self.num_state_clusters,
      len(self.stateHashToState)))
    for statehash,state in self.stateHashToState.items():
      dataset.append(state)
      a = self.best_action_from_statehash(statehash)[0]
      if a != -1:
        best_action.append(a)
    
    dataset = np.array(dataset)
    print("Dataset dimension : {}".format(dataset.shape))

    # Build KMeans clusters
    print("Fitting KMeans")
    self.kmeans = KMeans(n_clusters=self.num_state_clusters, max_iter=200, tol=0.0001, copy_x=True, n_jobs=4)
    self.kmeans.fit(dataset)
    clusters = self.kmeans.predict(dataset)

    # Collect most selected best actions from each cluster
    cluster_best_actions = {cid: [] for cid in range(self.num_state_clusters)} # [cluster_id => list[actionhash]]
    for c,a in zip(clusters, best_action):
      if best_action != -1:
        cluster_best_actions[c].append(a)

    def get_best_actions(cnt):
      tops = [i for i,freq in cnt.most_common(1)] #if i!=-1]
      if len(tops)==0:
        return -1
      else:
        return tops[0]

    # Take the best 2 actions to take for each cluster
    self.cluster_best_actions = {c: get_best_actions(Counter(ws))  \
      for c,ws in cluster_best_actions.items()}

    pop = Counter(clusters)
    for c, best_actions in self.cluster_best_actions.items():
      print("... Cluster #{} - {:4.0f} states => Best action : {}".format(
        c,
        pop[c],
        best_actions))

  def reset(self):
    """
    Reset all internal states
    """
    pass

  def learn(self, state, action, reward, next_state):
    """
    Learn that:
    - If we take an action (int) on the state (np.array)`
    - we will get back the reward (double value)
    - and register the next state (np.array)
    """
    pass

  def best_action_from_statehash(self, statehash):
    best_action = -1
    best_reward = 0
    for next_statehash in self.state_machine[statehash]:
      a = self.state_machine[statehash][next_statehash]
      v = self.get_v(next_statehash) or 0
      if v > best_reward:
        best_reward = v
        best_action = a
      return best_action, best_reward

  def best_action(self, state, silence=False):
    """
    Return the best action to take on the specified state 
    to maximise the possible reward
    """
    statevector,statehash = self.encoder.encode_state(state)

    if statehash not in self.state_machine:
      if self.kmeans is None:
        # Unrecognised state, return no recommended action
        print(colored("... Take random action on new state", "grey"))
        return (-1, 0)
      else:
        # Assume the closest state from experience
        closest = self.kmeans.predict([statevector])[0]
        action = self.cluster_best_actions[closest]

        if action==-1:
          print(colored("... Assume action from closest state", "blue"))
          return action, 0
        else:
          print(colored("... Take random action, not enough knowledge", "cyan"))
          return -1,0

    best_action, best_reward = self.best_action_from_statehash(statehash)

    if best_action == -1:
      if not silence:
        print(colored("... Relearn new action from experience", "yellow"))
    else:
      if not silence:
        print(colored("... Take best action from experience", "green"))
    return (best_action, best_reward)

  def get_v(self, statehash):
    """
    Evaluate the reward value of `state`
    """
    if statehash not in self.v:
      return None
    else:
      return self.v[statehash]

  def save(self, path):
    with open(path, "wb") as f:
      print("Saving the agent to {}".format(path))
      joblib.dump(self, f, compress=1)

  @staticmethod
  def load(path, default):
    if os.path.isfile(path):
      with open(path, "rb") as f:
        print("Agent loaded from {}".format(path))
        return joblib.load(f)
    else:
      print("No agent file to load, created a new one")
      return default


class TDAgent(Agent):
  """
  Temporal difference
  """
  def __init__(self, encoder=StateActionEncoder(), learning_rate=0.8, alpha=1.0):
    super().__init__(learning_rate)
    self.alpha = alpha
    self.encoder = encoder

  def learn(self, state, action, reward, next_state):
    statevec,statehash = self.encoder.encode_state(state)
    _,newstatehash     = self.encoder.encode_state(next_state)
    actionhash         = self.encoder.encode_action(action)

    old_v = self.get_v(statehash) or 0
    new_v = self.get_v(newstatehash) or old_v

    # Update state v matrix
    diff = self.learning_rate * (reward + self.alpha * new_v - old_v)
    self.v[statehash] = old_v + diff
    self.stateHashToState[statehash] = statevec

    # Update state transition
    if statehash not in self.state_machine:
      self.state_machine[statehash] = {}
    self.state_machine[statehash][newstatehash] = actionhash


class QAgent(Agent):
  """
  Simple Q-learning
  """
  pass

class PGAgent(Agent):
  """
  Policy Gradient
  """
  pass