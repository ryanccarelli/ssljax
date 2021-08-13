from optax import adagrad, adam, adamw, lamb, rmsprop


@Optimizer.register("adagrad")(adagrad)
@Optimizer.register("adam")(adam)
@Optimizer.register("adamw")(adamw)
@Optimizer.register("lamb")(lamb)
@Optimizer.register("rmsprop")(rmsprop)
