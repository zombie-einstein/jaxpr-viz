import typing

import jax
import pydot

from .dot import draw_dot_graph


def draw(f, collapse_primitives=True, show_avals=True) -> typing.Callable:
    """
    Visualise a JAX computation graph

    Wraps a JAX jit compiled function, which when called
    visualises the computation graph using
    pydot.

    Examples
    --------

    .. highlight:: python
    .. code-block:: python

       import jax
       import jpviz

       @jax.jit
       def foo(x):
           return 2 * x

       @jax.jit
       def bar(x):
           x = foo(x)
           return x - 1

       g = jpviz.draw(bar)(jax.numpy.arange(10))

    Parameters
    ----------
    f:
        JAX jit compiled function
    collapse_primitives: bool
        If
    show_avals: bool
        If `True` then type information will be
        included on node labels

    Returns
    -------
    Wrapped function that when called with concrete
    values generated the corresponding visualisation
    of the computation graph
    """

    def _inner_draw(*args, **kwargs) -> pydot.Graph:
        jaxpr = jax.make_jaxpr(f)(*args, **kwargs)
        return draw_dot_graph(jaxpr, collapse_primitives, show_avals)

    return _inner_draw


def view_pydot(dot_graph: pydot.Dot) -> None:
    """
    Show a pydot graph in a jupyter notebook

    Parameters
    ----------
    dot_graph: Graph
        Pydot graph as generated by `draw`
    """
    from IPython.display import Image, display

    plt = Image(dot_graph.create_png())
    display(plt)