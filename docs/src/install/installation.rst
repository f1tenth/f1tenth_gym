Installation
=================
``f1tenth_gym`` is a pure Python library and can be installed with ``pip``.


PyPI
------

Coming Soon...


Install for Developement
----------------------------

We suggest using a ``virtualenv``. Note that ``python3.9+`` is required. You can install the package via pip:

.. code:: bash

    pip install -e git+https://github.com/f1tenth/f1tenth_gym.git@v1.0.0#egg=f1tenth_gym


Building Documentation
------------------------

The documentations can be build locally. Install the extra dependencies via pip:

.. code:: bash

    cd f1tenth_gym/docs
    pip install -r requirements.txt

Then you can build the documentations in the ``docs/`` directory via:

.. code:: bash
    
    make html

The static HTML pages can be found at ``docs/_build/html/index.html`` afterwards.