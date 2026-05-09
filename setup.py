from setuptools import setup, find_packages

setup(
    name="pmiquest",
    version="1.0.0",
    description="PMI-QuEST: PMI-Augmented Query-by-Example Spoken Term Detection",
    author="Akanksha Singh",
    packages=find_packages(exclude=["experiments", "figures", "configs", "data", "results"]),
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.24",
        "scipy>=1.10",
        "scikit-learn>=1.3",
        "hnswlib>=0.7",
        "torch>=2.0",
        "torchaudio>=2.0",
        "transformers>=4.38",
        "soundfile>=0.12",
        "tqdm>=4.66",
        "matplotlib>=3.8",
        "pyyaml>=6.0",
    ],
    extras_require={
        "dev": ["pytest", "black", "isort"],
    },
)
