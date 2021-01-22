.. image:: assets/f1_stickers_01.png
  :width: 60
  :align: left

F1TENTH Gym Documentation 
================================================

This is the documentation of the F1TENTH Gym environment.

Citing
--------
If you find this Gym environment useful, please consider citing:

.. code::
  
  @inproceedings{o2020textscf1tenth,
    title={textscF1TENTH: An Open-source Evaluation Environment for Continuous Control and Reinforcement Learning},
    author={O’Kelly, Matthew and Zheng, Hongrui and Karthik, Dhruv and Mangharam, Rahul},
    booktitle={NeurIPS 2019 Competition and Demonstration Track},
    pages={77--89},
    year={2020},
    organization={PMLR}
  }

Physical Platform
-------------------

To build a physical 1/10th scale vehicle, following the guide here: https://f1tenth.org/build.html

.. image:: assets/f110cover.png
  :width: 400
  :align: center

.. toctree::
   :caption: INSTALLATION
   :maxdepth: 2

   installation


.. toctree::
   :caption: USAGE
   :maxdepth: 2

   basic_usage
   customized_usage

.. toctree::
   :caption: API REFERENCE
   :maxdepth: 2

   api/base_classes
   api/dynamic_models
   api/laser_models
   api/collision_models
   api/env
   api/obv