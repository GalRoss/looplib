import numpy as np
from setuptools import setup
from Cython.Build import cythonize

# The setup script
setup(
    name="looplib_bacterial_gal",
    version="0.1",
    install_requires=['numpy'],
    description="looplib by Anton Goloborodko, edited for circular bacterial chromosomes, added observables for MaxCal",
    packages=["looplib_bacterial_gal"],
    # Which files to cythonize
    ext_modules=cythonize(['looplib_bacterial_gal/bacterial_no_bypassing.pyx','looplib_bacterial_gal/bacterial_bypassing.pyx']),
    include_dirs=[np.get_include()]
)
