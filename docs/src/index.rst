F1TENTH Gym Documentation 
================================================

Overview
---------
The F1TENTH Gym environment is created for research that needs a asynchronous, realistic vehicle simulation with multiple vehicle instances in the same environment, with applications in reinforcement learning.

The environment is designed with determinism in mind. All agents' physics simulation are stepped simultaneously, and all randomness are seeded and experiments can be reproduced. The explicit stepping also enables the physics engine to take advantage of faster than real-time execution (up to 30x realtime) and enable massively parallel applications.

Github repo: https://github.com/f1tenth/f1tenth_gym

Note that the GitHub will have more up to date documentation than this page. If you see a mistake, please contribute a fix!


.. grid:: 2
  :gutter: 4

  .. grid-item-card::

    Physical Platform
    ^^^^^^^^^^^^^^^^^
    
    To build a physical 1/10th scale vehicle, follow the official build guide.
    
    .. image:: assets/f110cover.png
      :width: 100
      :align: center
    
    +++

    .. button-link:: https://f1tenth.org/build.html
      :expand:
      :color: secondary
      :click-parent:

      Build Guide


  .. grid-item-card::

    Installation
    ^^^^^^^^^^^^
    
    Installation guide

    .. image:: assets/pip_logo.svg
      :width: 100
      :align: center
    
    +++

    .. button-ref:: install/installation
      :expand:
      :color: secondary
      :click-parent:

      Installation

  .. grid-item-card::

    Quick Start
    ^^^^^^^^^^^
    
    Example usage

    .. image:: assets/gym.svg
      :width: 100
      :align: center
    
    +++

    .. button-ref:: usage/index
      :expand:
      :color: secondary
      :click-parent:
      
      Quick Start

  .. grid-item-card::

    API
    ^^^
    
    API
    
    .. image:: assets/gym.svg
      :width: 100
      :align: center

    +++

    .. button-ref:: api/index
      :expand:
      :color: secondary
      :click-parent:
      
      API

Citing
--------
If you find this Gym environment useful, please consider citing:

.. code::
  
  @inproceedings{o2020f1tenth,
    title={textscF1TENTH: An Open-source Evaluation Environment for Continuous Control and Reinforcement Learning},
    author={O’Kelly, Matthew and Zheng, Hongrui and Karthik, Dhruv and Mangharam, Rahul},
    booktitle={NeurIPS 2019 Competition and Demonstration Track},
    pages={77--89},
    year={2020},
    organization={PMLR}
  }

.. toctree::
  :hidden:
  :maxdepth: 3
  :titlesonly:

  install/installation
  usage/index
  api/index