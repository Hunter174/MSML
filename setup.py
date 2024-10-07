import os
import re
import subprocess
import sys
from pathlib import Path
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext

# A CMakeExtension builds the extension via a CMake process
class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=""):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)

class CMakeBuild(build_ext):
    def run(self):
        for ext in self.extensions:
            self.build_extension(ext)

    def build_extension(self, ext):
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))
        cfg = 'Debug' if self.debug else 'Release'
        cmake_args = [
            f'-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir}',
            f'-DPYTHON_EXECUTABLE={sys.executable}',
            f'-DCMAKE_BUILD_TYPE={cfg}',
        ]

        # Create build directory
        build_temp = self.build_temp
        if not os.path.exists(build_temp):
            os.makedirs(build_temp)

        # Configure and build with CMake
        subprocess.check_call(['cmake', ext.sourcedir] + cmake_args, cwd=build_temp)
        subprocess.check_call(['cmake', '--build', '.'], cwd=build_temp)

setup(
    name='linear_regression',
    version='0.1',
    author='Your Name',
    author_email='your_email@example.com',
    description='A simple linear regression module with C++/pybind11',
    ext_modules=[CMakeExtension('linear_regression')],
    cmdclass={'build_ext': CMakeBuild},
    zip_safe=False,
)
