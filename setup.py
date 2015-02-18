#!/usr/bin/env python
from setuptools import setup

setup(name="multipool",
      version="0.10.1",
      description="Efficient multi-locus genetic mapping with pooled sequencing.",
      author="Matt Edwards",
      author_email="matted@mit.edu",
      license="MIT",
      url="https://github.com/matted/multipool",
      packages=[],
      scripts=["mp_inference.py"],
      zip_safe=True,
      install_requires=["scipy", "numpy"], # pylab is optional; leaving it out for now
  )
