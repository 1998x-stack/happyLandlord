#!/usr/bin/env python
"""
Setup script for Happy Landlord 2V2 Reinforcement Learning Project
"""

from setuptools import setup, find_packages
import os

# Read the contents of README file
this_directory = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

# Read requirements
with open('requirements.txt') as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]

setup(
    name="happy-landlord",
    version="1.0.0",
    author="Happy Landlord Development Team",
    author_email="happy-landlord@example.com",
    description="A complete reinforcement learning implementation for Tencent's Happy Landlord 2V2 game",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    install_requires=requirements,
    python_requires='>=3.7',
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Games/Entertainment",
    ],
    keywords="reinforcement-learning, game-ai, deep-learning, landlord, gaming",
    license="MIT",
    url="https://github.com/example/happy-landlord",
    project_urls={
        "Documentation": "https://github.com/example/happy-landlord/blob/main/USER_GUIDE.md",
        "Source": "https://github.com/example/happy-landlord",
    },
    entry_points={
        'console_scripts': [
            'happy-landlord-train=happylandlord.main:main',
        ],
    },
    include_package_data=True,
    zip_safe=False,
)