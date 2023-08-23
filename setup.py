from setuptools import setup

setup(
    name="f110_gym",
    version="0.2.1",
    author="Hongrui Zheng",
    author_email="billyzheng.bz@gmail.com",
    url="https://f1tenth.org",
    package_dir={"": "gym"},
    install_requires=[
        "gymnasium",
        "numpy<=1.25.0,>=1.18.0",
        "Pillow>=9.0.1",
        "scipy>=1.7.3",
        "numba>=0.55.2",
        "pyyaml>=5.3.1",
        "pyglet<1.5",
        "pyopengl",
        "yamldataclassconfig",
        "requests",
        "shapely",
        "opencv-python",
    ],
    extras_require={
        'dev': [
            'pytest',
            'flake8',
            'black',
            'ipykernel',
            'isort',
            'autoflake',
            'matplotlib'
        ]
    }
)
