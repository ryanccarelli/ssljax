# ssljax 
Welcome to ssljax! We are implementing a framework for self-supervised representation learning
in Jax (with Flax and Optax). Our design philosophy prioritizes **flexibility** and **rapid iteration**.

View **link documentation**.

We would like to provide configurations for popular models, to benchmark and as starting points for researchers. 
Below find our (ambitious) wishlist.
### Model wishlist: 
1. DINO https://arxiv.org/abs/2104.14294
2. (TODO) ESVIT https://arxiv.org/abs/2106.09785
3. (TODO) SEER https://arxiv.org/abs/2103.01988
4. (TODO) MoCo https://arxiv.org/abs/2003.04297
5. (TODO) MoCov2 https://arxiv.org/abs/2003.04297
6. (TODO) MoCov3
7. (TODO) SimCLR https://arxiv.org/abs/2002.05709
8. BYOL https://arxiv.org/abs/2006.07733
9. (TODO) SwAV https://arxiv.org/abs/2006.09882
10. (TODO) Barlow Twins https://arxiv.org/abs/2103.03230
11. (TODO) PAWS https://arxiv.org/abs/2104.13963
12. (TODO) MAE https://arxiv.org/abs/2111.06377
13. (TODO) BEiT https://arxiv.org/pdf/2106.08254.pdf 


### Installation (from source)
````
git clone https://github.com/ryanccarelli/ssljax
cd ssljax 
conda env create --name ssljax python=3.8
conda activate ssljax
pip install -e .
````

### Installation (from pip)
````
conda env create --name ssljax python=3.8
pip install ssljax
````

### Contributing
```ssljax``` is an open source project, contribute to self-supervised learning in the Jax ecosystem.
