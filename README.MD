# Reinforcement learning climber
This project is inspired by [rock climbing](https://en.wikipedia.org/wiki/Rock_climbing).
The goal is to train a virtual climber to navigate a climbing route using [Q-learning](https://en.wikipedia.org/wiki/Q-learning), a model-free reinforcement learning algorithm.

## Project Overview
- **Route Definition**: A route is a limmited area with holds and [quickdraws](https://en.wikipedia.org/wiki/Quickdraw). The route is considered complete when the climber reaches the top hold.
- **Climber Model**: The climber is a simple model with limbs and basic joints. It can move only by using the holds and must clip the quickdraws, meaning it has to pass the rope through them.
- **Climbing Technique**: The climber uses a technique involving alternating support between opposing limbs. This means most of the climber's weight is supported by the right arm and left leg, or the left arm and right leg. The other limbs are moved to the next holds.

## Prerequisites
**Python 3.8** is required and dependencies from [requirements.txt](./requirements.txt)
or [Pipfile.lock](./Pipfile.lock).

An example of setting up the environment on Fedora Linux.

Install python 3.8:
```shell
dnf install python3.8
```

Install pip:
```shell
python3.8 -m ensurepip --upgrade
```

Install dependencies:
```shell
python3.8 -m pip install -r requirements.txt
```

## Running
To run the project, execute the following command:
```shell
python3.8 ./do_q_learning.py
```
This command launches the GUI with the climber. The model will start learning, with progress displayed in the console (this may take a few minutes). The climber will then attempt to reach the top hold or continue until the move limit is exceeded. Finally, a graph displaying the mean rewards will be shown.