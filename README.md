# The F1TENTH Gym environment with Google Colab

This is the repository of the [F1TENTH](https://f1tenth.org/) Gym environment with [Google Colaboratory](https://research.google.com/colaboratory/) integration.

#### F1TENTH Gym with Reinforcement Learning implementation:
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/avantgarda/f1tenth_gym/blob/colab/colab/F1TenthGymRL.ipynb)
#### F1TENTH Gym basic integration with Colab:
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/avantgarda/f1tenth_gym/blob/colab/colab/F1TenthGym.ipynb)

This project is still under heavy developement.

You can find the [documentation](https://f1tenth-gym.readthedocs.io/en/latest/) of the environment here.

## Quickstart
You can install the environment by running:

### Google Colab:

```bash
!git clone https://github.com/avantgarda/f1tenth_gym.git # cloning from avantgarda fork of the F1Tenth repo
%cd /content/f1tenth_gym
!git checkout colab # colab-compatible branch
%cd /content/f1tenth_gym/gym
!python setup.py install
```

### Local machine:

```bash
$ git clone https://github.com/f1tenth/f1tenth_gym.git
$ cd f1tenth_gym
$ git checkout exp_py
$ pip3 install --user -e gym/
```

Then you can run a quick waypoint follow example by:
```bash
cd examples
python3 waypoint_follow.py
```

## Known issues
- On MacOS Big Sur and above, when rendering is turned on, you might encounter the error:
```
ImportError: Can't find framework /System/Library/Frameworks/OpenGL.framework.
```
You can fix the error by installing a newer version of pyglet:
```bash
$ pip3 install pyglet==1.5.11
```
And you might see an error similar to
```
gym 0.17.3 requires pyglet<=1.5.0,>=1.4.0, but you'll have pyglet 1.5.11 which is incompatible.
```
which could be ignored. The environment should still work without error.

## Citing
If you find this Gym environment useful, please consider citing:

```
@inproceedings{okelly2020f1tenth,
  title={F1TENTH: An Open-source Evaluation Environment for Continuous Control and Reinforcement Learning},
  author={O’Kelly, Matthew and Zheng, Hongrui and Karthik, Dhruv and Mangharam, Rahul},
  booktitle={NeurIPS 2019 Competition and Demonstration Track},
  pages={77--89},
  year={2020},
  organization={PMLR}
}
```
