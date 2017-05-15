#!/usr/bin/env python

from __future__ import print_function
from distutils.core import setup
from distutils.command.install import install as DistutilsInstall  # noqa
import sys
import subprocess


try:
    import tensorflow as tf  # noqa
except ImportError:
    print("Please install tensorflow 0.12.0 or later")
    sys.exit()


class Makefile(DistutilsInstall):
    def run(self):
        subprocess.call(['make', '-C', 'matrix_add_op', 'build'])
        DistutilsInstall.run(self)

setup(name='matrix-add',
      version='0.1',
      description='A simple example for adding custom Ops to TensorFlow',
      packages=['matrix_add'],
      package_data={'matrix_add': ['matrix_add_op.so']},
      cmdclass={'install': Makefile})
