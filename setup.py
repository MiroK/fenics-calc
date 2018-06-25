#!/usr/bin/env python

from distutils.core import setup

setup(name = 'fenics_calc',
      version = '0.1',
      description = 'Lazy calculator over FEniCS functons',
      author = 'Miroslav Kuchta',
      author_email = 'miroslav.kuchta@gmail.com',
      url = 'https://github.com/mirok/fenics-calc.git',
      packages = ['xcalc'],
      package_dir = {'xcalc': 'xcalc'}
)
