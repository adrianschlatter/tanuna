# -*- coding: utf-8 -*-
"""
@author: Adrian Schlatter
"""

from setuptools import setup

setup(name='tanuna',
      version='0.0',
      description='Python tools to work with dynamic systems',
      url='https://github.com/adrianschlatter/tanuna',
      author='Adrian Schlatter',
      author_email='schlatter@phys.ethz.ch',
      license='Revised BSD',
      packages=['tanuna'],
      install_requires=['numpy', 'scipy'],
      include_package_data=True,
      test_suite='tests',
      zip_safe=False)
