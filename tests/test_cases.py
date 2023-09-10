import jax
import jax.numpy as jnp


@jax.jit
def func1(first, second):
    temp = first + jnp.sin(second) * 3.0
    return jnp.sum(temp)


def func2(inner, first, second):
    temp = first + inner(second) * 3.0
    return jnp.sum(temp)


def inner_func(second):
    if second.shape[0] > 4:
        return jnp.sin(second)
    else:
        assert False


@jax.jit
def func3(first, second):
    return func2(inner_func, first, second)


@jax.jit
def func4(arg):
    temp = arg[0] + jnp.sin(arg[1]) * 3.0
    return jnp.sum(temp)


@jax.jit
def one_of_three(index, arg):
    return jax.lax.switch(
        index, [lambda x: x + 1.0, lambda x: x - 2.0, lambda x: x + 3.0], arg
    )


@jax.jit
def func7(arg):
    return jax.lax.cond(
        arg >= 0.0, lambda x_true: x_true + 3.0, lambda x_false: x_false - 3.0, arg
    )


@jax.jit
def func8(arg1, arg2):
    return jax.lax.cond(
        arg1 >= 0.0,
        lambda x_true: x_true[0],
        lambda x_false: jnp.array([1]) + x_false[1],
        arg2,
    )


@jax.jit
def func10(arg, n):
    ones = jnp.ones(arg.shape)
    return jax.lax.fori_loop(
        0, n, lambda i, carry: carry + ones * 3.0 + arg, arg + ones
    )


@jax.jit
def func11(arr, extra):
    ones = jnp.ones(arr.shape)

    def body(carry, a_elems):
        ae1, ae2 = a_elems
        return carry + ae1 * ae2 + extra, carry

    return jax.lax.scan(body, 0.0, (arr, ones))


test_cases = [
    (func1, [jnp.zeros(8), jnp.ones(8)]),
    (func3, [jnp.zeros(8), jnp.ones(8)]),
    (func4, [(jnp.zeros(8), jnp.ones(8))]),
    (one_of_three, [1, 5.0]),
    (func7, [5.0]),
    (func8, [5.0, (jnp.zeros(1), 2.0)]),
    (func10, [jnp.ones(16), 5]),
    (func11, [jnp.ones(16), 5.0]),
]
