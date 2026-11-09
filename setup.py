from setuptools import find_packages, setup

setup(name='f110_gym',
      version='0.2.1',
      author='Hongrui Zheng',
      author_email='billyzheng.bz@gmail.com',
      url='https://f1tenth.org',
      package_dir={'': 'gym'},
      packages=find_packages(where='gym'),
      package_data={'f110_gym': ['envs/maps/*.yaml', 'envs/maps/*.png',
                                 'envs/maps/*.pgm']},
      python_requires='>=3.11,<3.16',
      # Keep the legacy Gym API and pyglet renderer; newer major versions
      # change the reset/step contract and OpenGL drawing API respectively.
      install_requires=['gym==0.26.2',
                        'numpy>=2.3,<3',
                        'Pillow>=12',
                        'scipy>=1.16.1,<2',
                        'numba>=0.68,<0.69',
                        'pyyaml>=6.0.3',
                        'pyglet>=1.5.31,<2',
                        'pyopengl>=3.1.10']
      )
