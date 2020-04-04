# Gym Reinforce

Experiment panel for trying reinforcement learning algorithms 
with some tricks.

## Prerequisites

You need [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or 
typical [Anaconda](https://www.anaconda.com/) to setup an environment.

```bash
  $ conda env create -f environment.yml
  $ conda activate gym
```

## Run

A sample car racing game, spun off from https://github.com/frankfurt-hackathon/carracing-aigym 
is implemented with Temporal-Difference learning. This can be run as follows.

```bash
  $ python3 -m game.carrace
```

Gym will start up a simulation environment and `game` module will govern the 
learning loop. See `model/` for the output agent and its learning log.


### Licence

BSD-2