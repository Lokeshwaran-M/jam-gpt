import os
from setuptools import setup, find_packages

setup(
    name='jam-gpt',
    version='0.0.4',
    description='A reimplementation of large language model (LLM) architectures designed for research and development processes',
    author='Lokeshwaran M',
    author_email='lokeshwaran.m23072003@gmail.com',
    url="https://github.com/Lokeshwaran-M/jam-gpt.git",
    license="MIT",
    packages=find_packages(),
    package_data={'': ['requirements.txt', 'README.md']},
    install_requires=open('requirements.txt').readlines(),
    keywords='jam-gpt Jam-AI Jam-AGI',
)


# install_requires=["setuptools==67.8.0","torch==2.0.1","tiktoken"]

