from setuptools import setup, find_packages

setup(
    name="feedtransformation",
    version="0.0.2",
    description="A Python package that leverages the power of Polars to efficiently group and create nested JSON product feed files. Ideal for handling large datasets and complex JSON structures.",
    author="José Carlos Borrayo Tojín",
    author_email="thecircuitproject1@gmail.com",
    packages=find_packages(),
    install_requires=["polars>=0.20.14"],
)
