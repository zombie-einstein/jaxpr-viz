# Gallery

## Simple Function

```python
@jax.jit
def func1(first, second):
    temp = first + jnp.sin(second) * 3.0
    return jnp.sum(temp)
```

![func1](.github/images/gallery/func1.png)

## One of Three (Switch)

`collapse_primitives=False`

```python
@jax.jit
def one_of_three(index, arg):
    return jax.lax.switch(
        index,
        [lambda x: x + 1.0, lambda x: x - 2.0, lambda x: x + 3.0],
        arg
    )
```

![one_of_three](.github/images/gallery/one_of_three.png)

## Binary Switch

`collapse_primitives=False`

```python
@jax.jit
def func7(arg):
    return jax.lax.cond(
        arg >= 0.0,
        lambda x_true: x_true + 3.0,
        lambda x_false: x_false - 3.0,
        arg
    )
```

![func7](.github/images/gallery/func7.png)

## Switch with Tuple

`collapse_primitives=True`

```python
@jax.jit
def func8(arg1, arg2):
    return jax.lax.cond(
        arg1 >= 0.0,
        lambda x_true: x_true[0],
        lambda x_false: jnp.array([1]) + x_false[1],
        arg2,
    )
```

![func8](.github/images/gallery/func8.png)

## For i Loop

`collapse_primitives=True`

```python
@jax.jit
def func10(arg, n):
    ones = jnp.ones(arg.shape)
    return jax.lax.fori_loop(
        0, n, lambda i, carry: carry + ones * 3.0 + arg, arg + ones
    )
```

![func10](.github/images/gallery/func10.png)

## Scan

`collapse_primitives=False`

```python
@jax.jit
def func11(arr, extra):
    ones = jnp.ones(arr.shape)

    def body(carry, a_elems):
        ae1, ae2 = a_elems
        return carry + ae1 * ae2 + extra, carry

    return jax.lax.scan(body, 0.0, (arr, ones))
```

![func11](.github/images/gallery/func11.png)
