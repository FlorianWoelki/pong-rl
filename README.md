# pong-rl

This repository contains the code for the Pong RL project.
The goal of this project is to train an agent to play the game of Pong using reinforcement learning.
The agent is trained using the (REINFORCE)[https://link.springer.com/content/pdf/10.1007/BF00992696.pdf] algorithm.

## Installation

To install the required packages, run the following command:

```bash
pip install -r requirements.txt
```

## Usage

To train the agent, run the following command:

```bash
python main.py
```

To generate a gif of the agent playing, run the following command:

```bash
python main.py -m animation
```

To see the agent play, run the following command:

```bash
python main.py -m render
```
