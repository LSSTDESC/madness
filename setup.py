"""Setup file."""

from setuptools import setup

with open("README.md") as readme_file:
    readme = readme_file.read()

requirements = [
    "galcheat",
    "matplotlib",
    "numpy",
    "pandas",
    "tensorflow",
    "tensorflow-probability",
    "scikit-image",
    "sep",
]

setup(
    name="maddeb",
    version="0.0.1",
    author="Biswajit Biswas",
    author_email="biswas@apc.in2p3.fr",
    # maintainer="Biswajit Biswas",
    # maintainer_email="biswajit.biswas@apc.in2p3.fr",
    description="Galaxy deblender from variational autoencoders",
    long_description=readme,
    long_description_content_type="text/markdown",
    url="https://github.com/b-biswas/MADNESS",
    include_package_data=True,
    packages=["maddeb"],
    install_requires=requirements,
    classifiers=[
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
    package_data={"maddeb": ["data/*"]},
)
