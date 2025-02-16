from setuptools import setup

setup(
    name='weap_util',
    version='0.1.0',
    author='Aly Ashour',
    package_dir={'': '.'},
    install_requires=['numpy<=1.22.0,>=1.18.0', 'opencv-python-headless==4.11.0.86']
)