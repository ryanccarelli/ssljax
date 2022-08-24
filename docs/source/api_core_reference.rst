Core API
========

Task
----

Every ``ssljax`` configuration defines a ``Task``.

.. autoapiclass:: ssljax.train.task.Task


Train
-----

The core ssljax class implementing the self-supervised training loop.

.. autoapiclass:: ssljax.train.trainer.Trainer

.. autoapiclass:: ssljax.train.ssltrainer.SSLTrainer

.. autoapiclass:: ssljax.train.trainstate.TrainState


Branch
------

.. autoapiclass:: ssljax.models.branch.branch.Branch

Pipelines
---------

Pipelines implement data augmentation.

.. autoapiclass:: ssljax.augment.pipeline.pipeline.Pipeline

Augmentations
-------------

.. autoapiclass:: ssljax.augment.augmentation.augmentation.Augmentation
.. autoapiclass:: ssljax.augment.augmentation.augmentation.AugmentationDistribution

We provide implementations of common augmentations.

.. autoapiclass:: ssljax.augment.augmentation.augmentation.RandomFlip
.. autoapiclass:: ssljax.augment.augmentation.augmentation.GaussianBlur
.. autoapiclass:: ssljax.augment.augmentation.augmentation.ColorTransform
.. autoapiclass:: ssljax.augment.augmentation.augmentation.Solarize
.. autoapiclass:: ssljax.augment.augmentation.augmentation.Clip
.. autoapiclass:: ssljax.augment.augmentation.augmentation.Identity
.. autoapiclass:: ssljax.augment.augmentation.augmentation.RandomCrop
.. autoapiclass:: ssljax.augment.augmentation.augmentation.CenterCrop

Models
------

.. autoapiclass:: ssljax.models.model.Model
.. autoapiclass:: ssljax.models.vit.ViT
.. autoapiclass:: ssljax.models.resnet.ResNet
.. autoapiclass:: ssljax.models.mlp.MLP
.. autoapiclass:: ssljax.models.mixer.Mixer

Postprocess
-----------

.. autoapiclass:: ssljax.train.postprocess.postprocess.PostProcess
.. autoapifunction:: ssljax.train.postprocess.ema.ema_builder

Optimizers
----------
We wrap optimizers from Deepmind's `optax <https://github.com/deepmind/optax>`_ library.

    ================================ ==============================================================================================================================================================
    Optimizer                        Reference
    ================================ ==============================================================================================================================================================
    ``ssljax.optimizers.adabelief``  `arxiv <https://arxiv.org/abs/2010.07468>`_
    ``ssljax.optimizers.adagrad``    `jmlr <https://jmlr.org/papers/v12/duchi11a.html>`_
    ``ssljax.optimizers.adam``       `arxiv <https://arxiv.org/abs/1412.6980>`_
    ``ssljax.optimizers.adamw``      `arxiv <https://arxiv.org/abs/1711.05101>`_
    ``ssljax.optimizers.dpsgd``      `arxiv <https://arxiv.org/abs/1607.00133>`_
    ``ssljax.optimizers.fromage``    `arxiv <https://arxiv.org/pdf/2002.03432>`_
    ``ssljax.optimizers.lamb``       `arxiv <https://arxiv.org/abs/1904.00962>`_
    ``ssljax.optimizers.noisy_sgd``  `arxiv <https://arxiv.org/pdf/1911.11607>`_
    ``ssljax.optimizers.radam``      `arxiv <https://arxiv.org/abs/1908.03265>`_
    ``ssljax.optimizers.rmsprop``    `arxiv <https://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf>`_
    ``ssljax.optimizers.sgd``        `arxiv <https://projecteuclid.org/journals/annals-of-mathematical-statistics/volume-22/issue-3/A-Stochastic-Approximation-Method/10.1214/aoms/1177729586.full>`_
    ``ssljax.optimizers.yogi``       `arxiv <https://proceedings.neurips.cc/paper/2018/file/90365351ccc7437a1309dc64e4db32a3-Paper.pdf>`_
    ``ssljax.optimizers.lars``       `arxiv <https://arxiv.org/abs/1708.03888>`_
    ``ssljax.optimizers.zerog``      used internally in optax to compose optimizers
    ================================ ==============================================================================================================================================================


Schedulers
----------
We wrap schedulers from Deepmind's `optax <https://github.com/deepmind/optax>`_ library.

  | ``ssljax.scheduler.constant``
  | ``ssljax.scheduler.cosine_decay``
  | ``ssljax.scheduler.cosine_onecycle``
  | ``ssljax.scheduler.exponential_decay``
  | ``ssljax.scheduler.linear_onecycle``
  | ``ssljax.scheduler.piecewise_constant``
  | ``ssljax.scheduler.piecewise_interpolate``
  | ``ssljax.scheduler.polynomial``

We also implement ``ssljax.models.sslmodel.SSLModel``-specific schedulers.

.. autoapifunction:: ssljax.train.scheduler.scheduler.BYOLlars

Losses
------

.. autoapiclass:: ssljax.losses.loss.Losso

.. autoapifunction:: ssljax.losses.byol.byol_regression_loss

.. autoapifunction:: ssljax.losses.byol.byol_softmax_cross_entropy

Data
----

.. autoapiclass:: ssljax.data.dataloader.DataLoader

We provide dataloaders for popular datasets.
#TODO: Complete this section following refactor by Bun

.. autoapifunction:: ssljax.data.dataloader.MNISTLoader

Core
----

Register
^^^^^^^^

Tasks are constructed from config files by getting objects from a global registry.

.. autoapifunction:: ssljax.core.register.register
.. autoapifunction:: ssljax.core.register.get_from_register

Pytrees
^^^^^^^

We implement utilities for manipulating `pytrees <https://jax.readthedocs.io/en/latest/pytrees.html>`_.

.. autoapiclass:: ssljax.core.pytrees.ModelParamFilter
.. autoapifuntion:: ssljax.core.pytrees.add_prefix_to_dict_keys
.. autoapifuntion:: ssljax.core.pytrees.flattened_traversal

Utils
^^^^^

.. autoapifunction:: ssljax.core.utils.prepare_environment
