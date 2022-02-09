"""Generic entry point for Python application. Modified from Scenic.

This provides run() which performs some initialization and then calls the
provided main with a JAX PRNGKey and path to configuration.
We expect each project to have it's own main.py. It's very short but
makes it easier to maintain as the number of projects grows.

Usage in your main.py:
  from ssljax import app

  def main(rng: jnp.ndarray,
           config: str):
    # Call the library that trains your model.

  if __name__ == '__main__':
    app.run(main)
"""

import functools
import logging

import jax
import tensorflow as tf
from absl import app, flags
from clu import metric_writers, platform

FLAGS = flags.FLAGS

# These are general flags that are used across most of scenic projects. These
# flags can be accessed via `flags.FLAGS.<flag_name>` and projects can also
# define their own flags in their `main.py`.
flags.DEFINE_string("config", None, "Task configuration.")
flags.DEFINE_string("workdir", None, "Work unit directory.")
flags.mark_flags_as_required(["config", "workdir"])


def run(main):
    # Provide access to --jax_backend_target and --jax_xla_backend flags.
    jax.config.config_with_absl()
    app.run(functools.partial(_run_main, main=main))


def _run_main(argv, *, main):
    """Runs the `main` method after some initial setup."""
    del argv
    # Hide any GPUs form TensorFlow. Otherwise TF might reserve memory and make
    # it unavailable to JAX.
    tf.config.experimental.set_visible_devices([], "GPU")

    if FLAGS.jax_backend_target:
        logging.info("Using JAX backend target %s", FLAGS.jax_backend_target)
        jax_xla_backend = (
            "None" if FLAGS.jax_xla_backend is None else FLAGS.jax_xla_backend
        )
        logging.info("Using JAX XLA backend %s", jax_xla_backend)

    logging.info("JAX host: %d / %d", jax.process_index(), jax.process_count())
    logging.info("JAX devices: %r", jax.devices())

    # Add a note so that we can tell which task is which JAX host.
    # (task 0 is not guaranteed to be host 0)
    platform.work_unit().set_task_status(
        f"host_id: {jax.process_index()}, host_count: {jax.process_count()}"
    )

    rng = jax.random.PRNGKey(FLAGS.config.rng_seed)
    logging.info("RNG: %s", rng)

    main(rng=rng, config=FLAGS.config)
