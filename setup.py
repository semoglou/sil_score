from setuptools import setup, find_packages

setup(
    name="sil-score",
    version="0.1.6",
    description="Exact and approximate silhouette scoring with micro, macro, and weighted cluster averages.",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    author="Aggelos Semoglou",
    license="MIT",
    url="https://github.com/semoglou/sil_score",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.21",
        "scikit-learn>=1.0",
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
