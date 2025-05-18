from setuptools import setup

setup(name='f110_gym',
      version='0.3.0',
      author='UniBo Motorsport Driverless',
      author_email='driverless@motorsport.unibo.it',
      url='https://f1tenth.org',
      package_dir={'': 'gym'},
      install_requires=['gymnasium',
		                'numpy',
                        'Pillow',
                        'scipy',
                        'numba',
                        'pyyaml',
                        'pyglet<1.5',
                        'pyopengl']
      )
