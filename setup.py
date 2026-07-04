#!/usr/bin/env python
# Copyright (c) SEC-VCM Authors.
# Licensed under the MIT License.

import os
import subprocess
from setuptools import setup, find_packages
from setuptools.command.build_ext import build_ext


class CMakeBuildExt(build_ext):
    """Build the C++ entropy coding extension using CMake."""
    
    def run(self):
        for ext in self.extensions:
            self.build_cmake(ext)
    
    def build_cmake(self, ext):
        import glob
        import shutil
        
        ext_dir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))
        cmake_source_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'secvcm', 'cpp'))
        
        build_temp = os.path.join(self.build_temp, ext.name)
        os.makedirs(build_temp, exist_ok=True)
        
        cmake_args = [
            f'-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={ext_dir}',
            f'-DPYTHON_EXECUTABLE={subprocess.sys.executable}',
            '-DCMAKE_BUILD_TYPE=Release',
        ]
        
        build_args = ['--config', 'Release', '-j', str(os.cpu_count() or 4)]
        
        subprocess.check_call(['cmake', cmake_source_dir] + cmake_args, cwd=build_temp)
        subprocess.check_call(['cmake', '--build', '.'] + build_args, cwd=build_temp)


setup(
    name='sec-vcm',
    version='1.0.0',
    description='Symmetric Entropy-Constrained Video Coding for Machines (SEC-VCM)',
    author='SEC-VCM Authors',
    license='MIT',
    packages=find_packages(exclude=['train', 'scripts', 'third_party', 'tests']),
    python_requires='>=3.8',
    install_requires=[
        'numpy>=1.20.0',
        'torch>=1.10.0',
        'torchvision>=0.11.0',
        'pytorch-msssim>=0.2.0',
        'lpips',
        'pillow',
        'tqdm',
    ],
    ext_modules=[CMakeBuildExt('secvcm.entropy_models')],
    cmdclass={'build_ext': CMakeBuildExt},
)
