Core API
========

Task
----
Every ``ssljax`` experiment instantiates its components in a Task object.

.. autoapiclass:: ssljax.train.task.Task


SSLTrainer
----------
The core ssljax class implementing self-supervised training.

.. autoapiclass:: ssljax.train.ssltrainer.SSLTrainer


Branches
--------
.. autoapiclass:: ssljax.models.branch.branch.Branch

We implement branches for popular self-supervised learning models.

Bootstrap Your Own Latent
^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoapiclass:: ssljax.models.branch.byolbranch.BYOLOnlineBranch
.. autoapiclass:: ssljax.models.branch.byolbranch.BYOLTargetBranch

Self-Distillation with No Labels
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Postprocess
-----------

.. autoapiclass:: ssljax.train.postprocess.postprocess.PostProcess
.. autoapiclass:: ssljax.train.postprocess.ema.ema_builder

Optimizers
----------
We wrap optimizers from Deepmind's Optax library.

    ================================ ==============================================================================================================================================================
    Optimizer                        Reference
    ================================ ==============================================================================================================================================================
    ``ssljax.optimizers.adabelief``  .. _adabelief: https://arxiv.org/abs/2010.07468
    ``ssljax.optimizers.adagrad``    .. _adagrad: https://jmlr.org/papers/v12/duchi11a.html
    ``ssljax.optimizers.adam``       .. _adam: https://arxiv.org/abs/1412.6980
    ``ssljax.optimizers.adamw``      .. _adamw: https://arxiv.org/abs/1711.05101
    ``ssljax.optimizers.dpsgd``      .. _dpsgd: https://arxiv.org/abs/1607.00133
    ``ssljax.optimizers.fromage``    .. _fromage: https://arxiv.org/pdf/2002.03432
    ``ssljax.optimizers.lamb``       .. _lamb: https://arxiv.org/abs/1904.00962
    ``ssljax.optimizers.noisy_sgd``  .. _noisy_sgd: https://arxiv.org/pdf/1911.11607
    ``ssljax.optimizers.radam``      .. _radam: https://arxiv.org/abs/1908.03265
    ``ssljax.optimizers.rmsprop``    .. _rmsprop: https://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf
    ``ssljax.optimizers.sgd``        .. _sgd https://projecteuclid.org/journals/annals-of-mathematical-statistics/volume-22/issue-3/A-Stochastic-Approximation-Method/10.1214/aoms/1177729586.full
    ``ssljax.optimizers.yogi``       .. _yogi: https://proceedings.neurips.cc/paper/2018/file/90365351ccc7437a1309dc64e4db32a3-Paper.pdf
    ``ssljax.optimizers.lars``       .. _lars: https://arxiv.org/abs/1708.03888
    ``ssljax.optimizers.zerog``      used internally in optax to compose optimizers
    ================================ ==============================================================================================================================================================

Augmentations
-------------
.. autoapiclass:: ssljax.augment.augmentation.augmentation.Augmentation
.. autoapiclass:: ssljax.augment.augmentation.augmentation.AugmentationDistribution

We provide implementations of common augmentations.

Example Augmentations
^^^^^^^^^^^^^^^^^^^^^

.. autoapiclass:: ssljax.augment.augmentation.augmentation.RandomFlip
.. autoapiclass:: ssljax.augment.augmentation.augmentation.RandomGaussianBlur
.. autoapiclass:: ssljax.augment.augmentation.augmentation.ColorTransform
.. autoapiclass:: ssljax.augment.augmentation.augmentation.Solarize
.. autoapiclass:: ssljax.augment.augmentation.augmentation.Clip

Pipelines
---------
Augmentations are composed into Pipelines that transform data into branch inputs.

.. autoapiclass:: ssljax.augment.pipeline.pipeline.Pipeline

We provide implementations of popular pipelines.

Example Pipelines
^^^^^^^^^^^^^^^^^

.. autoapiclass:: ssljax.augment.pipeline.byolpipeline.BYOLOnlinePipeline
.. autoapiclass:: ssljax.augment.pipeline.byolpipeline.BYOLTargetPipeline



Schedulers
----------
We wrap schedulers from Deepmind's Optax library.

    =========================================== ==============================================================================================================================================================
    Scheduler                                   Reference
    =========================================== ==============================================================================================================================================================
    ``ssljax.scheduler.constant``
    ``ssljax.scheduler.cosine_decay``
    ``ssljax.scheduler.cosine_onecycle``
    ``ssljax.scheduler.exponential_decay``
    ``ssljax.scheduler.linear_onecycle``
    ``ssljax.scheduler.piecewise_constant``
    ``ssljax.scheduler.piecewise_interpolate``
    ``ssljax.scheduler.polynomial``
    =========================================== ==============================================================================================================================================================

Data
----

.. autoapiclass:: ssljax.data.dataloader.DataLoader

We provide dataloaders for popular datasets.

.. autoapifunction:: ssljax.data.dataloader.MNISTLoader

Utils
-----

Register
^^^^^^^^
Tasks are constructed from config files by getting objects from a globa registry.
.. autoapifunction:: ssljax.core.utils.register.register
