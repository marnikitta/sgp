import numpy
import setuptools
from Cython.Build import cythonize
from setuptools.extension import Extension

extensions = [Extension('sgp.sobol', ['sgp/sobol.pyx'], include_dirs=[numpy.get_include()])]

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="sgp",
    version="0.3.2",
    author="Nikita Marshalkin",
    author_email="marnikitta@gmail.com",
    description="Sparse gaussian process regression",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/marnikitta/sgp",
    packages=setuptools.find_packages(),
    ext_modules=cythonize(extensions),
    package_data={"": ["*.pyx"]},
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
