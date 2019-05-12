import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="sgp",
    version="0.2.1",
    author="Nikita Marshalkin",
    author_email="marnikitta@gmail.com",
    description="Sparse gaussian process regression",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/marnikitta/sgp",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
