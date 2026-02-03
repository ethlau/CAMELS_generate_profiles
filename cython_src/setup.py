import numpy as np
import os
import platform

# manipulate get_config_vars:
# 1. step: wrap functionality and filter
from distutils.sysconfig import get_config_vars as default_get_config_vars

def remove_pthread(x):
    if type(x) is str:
        # x.replace(" -pthread ") would be probably enough...
        # but we want to make sure we make it right for every input
        if x=="-pthread":
            return ""
        if x.startswith("-pthread "):
            return remove_pthread(x[len("-pthread "):])
        if x.endswith(" -pthread"):
            return remove_pthread(x[:-len(" -pthread")])
        return x.replace(" -pthread ", " ")
    return x

def my_get_config_vars(*args):
  result = default_get_config_vars(*args)
  # sometimes result is a list and sometimes a dict:
  if type(result) is list:
     return [remove_pthread(x) for x in result]
  elif type(result) is dict:
     return {k : remove_pthread(x) for k,x in result.items()}
  else:
     raise Exception("cannot handle type"+type(result))

# 2.step: replace    
import distutils.sysconfig as dsc
dsc.get_config_vars = my_get_config_vars


# 3.step: normal setup.py

from distutils.core import setup
from Cython.Build import cythonize

from distutils.extension import Extension

# Define the extension module
extension = Extension(
    "process_halos_cy",
    sources=["process_halos_cy.pyx"],
    include_dirs=[np.get_include()],
    #library_dirs=["/opt/homebrew/Caskroom/miniforge/base/lib"],
    #extra_link_args = [],
)

# Run the setup
setup(
    name="process_halos_cy",
    ext_modules=cythonize([extension])
)
