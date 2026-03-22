from setuptools import setup, find_packages

setup(
    name="nemesis-ct",
    version="0.1.0",
    description="Superpatch-based 3D Medical Image Self-Supervised Pretraining via Noise-Enhanced Dual-Masking",
    author="Hyeonseok Jung",
    url="https://github.com/whilethis00/NEMESIS-public",
    packages=find_packages(exclude=["benchmark*", "scripts*", "data*", "results*"]),
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.24.0",
        "einops>=0.7.0",
    ],
    extras_require={
        "benchmark": [
            "monai>=1.3.0",
            "nibabel>=5.0.0",
            "scipy>=1.10.0",
            "tqdm>=4.65.0",
        ],
        "full": [
            "monai>=1.3.0",
            "nibabel>=5.0.0",
            "scipy>=1.10.0",
            "tqdm>=4.65.0",
            "huggingface_hub>=0.20.0",
            "tensorboard>=2.13.0",
        ],
    },
)
