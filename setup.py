import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="ssljax",
    version="1.0.0",
    author="The AI Discord, et al",
    author_email="ryan_carelli@dfci.harvard.edu",
    description="Self-Supervised Learning in Jax",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(),
    project_urls={
        "Documentation": "https://ssljax.readthedocs.io/en/stable",
        "Source Code": "https://github.com/Dana-Farber-AIOS/pathml",
    },
    install_requires=["numpy", "pre-commit==2.13.0", "torch", "flax",],
    extras_require=["jax>=0.1.71", "jaxlib>=0.1.49",],
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
