# -*- coding: utf-8 -*-
"""
@author: Adrian Schlatter
"""

from setuptools import setup

setup(name='tanuna',
      version='0.1',
      description='Python tools to work with dynamic systems',
      url='https://github.com/adrianschlatter/tanuna',
      author='Adrian Schlatter',
      author_email='',
      license='Revised BSD',
      packages=['tanuna'],
      install_requires=['numpy', 'scipy'],
      include_package_data=True,
      zip_safe=False)
