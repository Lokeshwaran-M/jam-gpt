import os
from setuptools import setup, find_packages

setup(
    name='jam-gpt',
    version='0.1',
    description='A reimplementation of large language model (LLM) architectures designed for research and development processes',
    author='Lokeshwaran M',
    author_email='lokeshwaran.m23072003@gmail.com',
    url="https://github.com/Lokeshwaran-M/jam-gpt.git",
    license="MIT",
    packages=find_packages(),
    install_requires=open('requirements.txt').readlines(),
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'License :: MIT License',
        'Programming Language :: Python :: 3.11',
    ],
    keywords='Jam-AI jam-gpt',
)
