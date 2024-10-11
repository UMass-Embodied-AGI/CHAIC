from pathlib import Path
from setuptools import setup, find_packages

readme = Path('README.md').read_text(encoding='utf-8')

with open('requirements.txt') as f:
    required = f.read().splitlines()

setup(
    name='transport_challenge_multi_agent',
    version="0.3.0",
    description='High-Level Multi-Agent Transport Challenge API for CHAIC.',
    long_description=readme,
    long_description_content_type='text/markdown',
    author='CHAIC Team',
    keywords='unity simulation multiple robotics agents',
    packages=find_packages(),
    include_package_data=True,
    install_requires=required,
)
