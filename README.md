![Python 3.8 3.9](https://github.com/f1tenth/f1tenth_gym/actions/workflows/ci.yml/badge.svg)
![Docker](https://github.com/f1tenth/f1tenth_gym/actions/workflows/docker.yml/badge.svg)
# The F1TENTH Gym environment

This is the repository of the F1TENTH Gym environment.

You can find the [documentation](https://f1tenth-gym.readthedocs.io/en/latest/) of the environment here.

## General Setup
Make sure you have Git installed through [here](https://git-scm.com/downloads)

Once you have downloaded Git, you should be able to clone the repository onto your PC.
```ps
git clone https://github.com/WE-Autopilot/f1tenth_gym.git
```

## Windows Setup
We recommend installing the simulation inside a virtual environment to not affect any of your other Python installations.

You must make sure your execution policy allows for scripts to run automatically.

***IMPORTANT: Run Windows PowerShell as Administrator, otherwise the following two commands will not work.***

If the command below outputs anything other than RemoteSigned, Run the next command.
```ps
Get-ExecutionPolicy                                                                        
```
```ps
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned
```
Follow with the steps on the screen, you might have to type in "Y" to accept the changes.

Next, you must download pyenv. To do so, run the following command after ***RESTARTING*** PowerShell.
```ps
Invoke-WebRequest -UseBasicParsing -Uri "https://raw.githubusercontent.com/pyenv-win/pyenv-win/master/pyenv-win/install-pyenv-win.ps1" -OutFile "./install-pyenv-win.ps1"; &"./install-pyenv-win.ps1"
```

Then, you must download the following specific version of python through pyenv.
```ps
pyenv install 3.10.11
```

Navigate to your File Explorer to find the location of where pyenv is installed. It should be in a similar directory as below:
```ps
%USERPROFILE%\.pyenv\pyenv-win\versions\
```

For Example:
```ps
C:\Users\<YourUsername>\.pyenv\pyenv-win\versions\3.10.11
```

Now, we get to the good stuff...

Go to where you cloned your repository and navigate to the root directory of your workspace and open a terminal.

Run the following command within your directory to create a virtual environment with the previously installed Python version.

```ps
<Your_Path_Without_Quotes> -m venv .venv

#For Example:

C:\Users\<YourUsername>\.pyenv\pyenv-win\versions\3.10.11\python.exe -m venv .venv
```

Then, install all the dependencies by running the following:

```ps
pip install -e .
# This command will take a while, so don't ear, just let it run!
```

***You are almost done!!!***

Change Directory to the example Python file, and run that file:
```ps
cd .\examples\
```

```ps
python .\waypoint_follow.py
```

It will take a second to run, once it does you should be all good!

## MacOS/Linux Setup
We recommend installing the simulation inside a virtual environment to not affect any of your other Python installations.  

You can install the environment any way you like. Instructions to do it with virtualenv are below.

Make sure to use python version 3.10. We've tested 3.10.11 and 3.10.15 and both have worked.

```bash
virtualenv .venv
source .venv/bin/activate
```

Then clone the repo
```bash
git clone https://github.com/WE-Autopilot/f1tenth_gym.git
cd f1tenth_gym
```

Set some versions by hand to avoid magic, tracebackless errors.
```bash
pip install "pip<24.1"
pip install "setuptools==65.5.0"
pip install "wheel<0.40.0"
```

Then run the gym setup
```bash
pip install -e .
```

You can run a quick waypoint follow example by:
```bash
cd examples
python waypoint_follow.py
```

It will take a second to run, once it does you should be all good!

## Known issues
You might see an error similar to
```
f110-gym 0.2.1 requires pyglet<1.5, but you have pyglet 1.5.20 which is incompatible.
```
which could be ignored. The environment should still work without error.

## Questions?

Contact one of the execs or your leads on Discord and ask any questions and we will help ASAP