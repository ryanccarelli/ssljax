Self-Supervised Learning
========================

Recent breakthroughs in machine learning have been driven by scaling neural models.
But how do we scale deep learning systems in domains where curated and labeled data is limited?

In ssljax models, samples are processed by branches composed of a model body, projector,
and (optionally) predictor. Each branch is assigned a pipeline of augmentations, applied to
each sample before it is passed through the branch.

Here show how popular self-supervised learning models fit into our abstraction, and provide
bencharks Table of model performances (as implemented in SSLJax)

Early Examples, Pretext Tasks, NLP
==================================

SIMCLR: Contrastive Learning and Negative Pairs
===============================================

Motivated by promising results in contrastive representation learning :footcite:t:`2014:Dosovitskiy,2018:Oord,2019:Bachman`,
SimCLR :footcite:t:`2020:chen` was the first technique to match supervised ResNet50 performance
on ImageNet by training a linear classifier on self-supervised features.
When fine-tuned on 1% of labels, outperforms AlexNet





The Importance of Augmentation
------------------------------

.. footbibliography::
