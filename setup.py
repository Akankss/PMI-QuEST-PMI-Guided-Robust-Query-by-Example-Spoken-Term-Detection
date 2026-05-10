from setuptools import setup, find_packages

setup(
    name="pmiquest",
    version="1.0.0",
    description="PMI-Guided Robust Query-by-Example Spoken Term Detection",
    author="Akanksha Singh, Yi-Ping Phoebe Chen, Vipul Arora",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.21",
        "scipy>=1.7",
        "scikit-learn>=1.0",
        "hnswlib>=0.7.0",
    ],
)
