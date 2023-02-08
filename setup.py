"""
setup.py

Created by: Andrius Bernatavicius
On: 08/02/2023, 16:33
"""

from setuptools import setup

requirements = [
    'scipy',
    'meeko',
    'docker']


setup(
    name='vinagpu',
    version='0.0.1',
    description='VinaGPU - AutoDock Vina on GPU, using Docker',
    requires=requirements,
    packages=['vinagpu'],

)