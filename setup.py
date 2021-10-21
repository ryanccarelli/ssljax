import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="ssljax",
    version="1.0.0",
    author="Ryan Carelli, Akash Ganesan",
    author_email="ryancarelli@gmail.com",
    description="self-supervised learning in jax",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(),
    project_urls={
        "Documentation": "https://ssljax.readthedocs.io/en/stable",
        "Source Code": "https://github.com/ryanccarelli/ssljax",
    },
    install_requires=[
        "numpy",
        "pre-commit",
        "torch",
        "chex",
        "flax",
        "tqdm",
        "base58",
        "dill",
        "overrides",
        "torchvision",
        "hydra-core",
        "tensorflow",
        "tensorboardX",
        "tensorboard",
        "parameterized",
        "vision_transformer @ git+https://github.com/google-research/vision_transformer.git",
    ],
    extras_require={"jax": ["jax>=0.1.71", "jaxlib>=0.1.49",]},
    requires_python=">=3.8",
    classifiers=[
        "License :: OSI Approved :: GNU General Public License v2 (GPLv2)",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Operating System :: OS Independent",
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Framework :: Sphinx",
        "Framework :: Pytest",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Processing",
        "Topic :: Scientific/Engineering :: Image Recognition",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
)
