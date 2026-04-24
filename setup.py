"""
Install ELSA as a Python package named `elsa`.

    pip install -e .          # editable install (recommended for development)
    pip install .             # regular install

After installation, import with:

    import elsa
    from elsa import ELSA_triton, ElsaAttention
"""
from setuptools import setup, find_packages

setup(
    name="elsa-attention",
    version="1.0.0",
    description="ELSA: Exact Linear-Scan Attention for Fast and Memory-Light Vision Transformers",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    author="Chih-Chung Hsu, Xin-Di Ma, Wo-Ting Liao, Chia-Ming Lee",
    url="https://ming053l.github.io/ELSA_projectpage/",
    license="See LICENSE (Academic & Non-Commercial Use Only)",
    python_requires=">=3.9",
    package_dir={"elsa": "code/stable"},
    packages=["elsa"],
    install_requires=[
        "torch>=2.1",
        "triton>=2.2",
        "timm>=0.9",
    ],
    extras_require={
        "benchmark": [
            "numpy>=1.24",
            "pandas>=1.5",
            "matplotlib>=3.7",
            "seaborn>=0.12",
            "datasets>=2.14",
            "Pillow>=9.0",
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
