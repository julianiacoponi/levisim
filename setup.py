from setuptools import setup, find_packages

setup(
    name="levisim",
    version="0.0.1",
    author="Julian Iacoponi",
    author_email="julian.iacoponi@gmail.com",
    description="A package for levitated physics simulations",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    packages=find_packages(),
    python_requires=">=3.9",
    include_package_data=True,
)