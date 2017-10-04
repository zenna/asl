from setuptools import setup

setup(
    name='asl',
    version='0.0.1',
    description='A library for learning algebraic structures',
    author='Zenna Tavares',
    author_email="zenna@mit.edu",
    packages=['asl'],
    install_requires=['tensorflow>=1.3.0rc0',
                      'numpy>=1.7'],
    url='https://github.com/zenna/asl',
    license='Apache License 2.0',
    classifiers=['License :: OSI Approved :: Apache Software License',
                 'Programming Language :: Python :: 2.7',
                 'Programming Language :: Python :: 3.4'],
)
