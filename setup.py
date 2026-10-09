from setuptools import setup, Extension
import numpy

src = \
[
    f'./nudft_src/ext/main.cpp',
]

ext = Extension\
(
    f"nudft.ext", 
    sources = src,
    include_dirs = [f"./nudft_src/ext/", numpy.get_include()],
    language = 'c++',
    extra_compile_args = ["-O3", "-fopenmp"],
    extra_link_args = ["-fopenmp"],
)

setup\
(
    name = f'nudft',
    ext_modules = [ext],
    packages = ["nudft"],
    package_dir = {"nudft":"./nudft_src/"},
    include_package_data = False
)
