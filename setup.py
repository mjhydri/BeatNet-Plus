"""
BeatNet+: CRNN and Particle Filtering for Online Joint Beat, Downbeat, and Meter Tracking
"""

import setuptools
from setuptools import find_packages

REQUIRED_PACKAGES = [
    'numpy>=1.21.0',
    'cython',
    'librosa>=0.8.0',
    'scipy',
    'mido>=1.2.6',
    'pytest',
    'madmom>=0.16.1',
    'torch>=1.9.0',
    'Matplotlib',
    'tensorboard>=2.0',
    'pyyaml>=5.0',
]

setuptools.setup(
    name="BeatNetPlus",
    version="1.0.0",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    include_package_data=True,
    install_requires=REQUIRED_PACKAGES,
    author="Mojtaba Heydari",
    author_email="mheydari@ur.rochester.edu",
    description="BeatNet+: Enhanced real-time and offline music beat/downbeat/tempo/meter tracking with multi-step training",
    keywords="Beat tracking, Downbeat tracking, meter detection, tempo tracking, particle filtering, source separation, singing voice, training",
    url="https://github.com/mjhydri/BeatNet",
    python_requires='>=3.8',
)