import pydot
from jax._src import core as jax_core

from . import graph


def draw_dot_graph(
    x: jax_core.ClosedJaxpr, collapse_primitives: bool, show_avals: bool
) -> pydot.Graph:
    """
    Generate a pydot representation of an XLA graph

    Parameters
    ----------
    x : ClosedJaxpr
    collapse_primitives: bool
        If `True` functions that are composed of only primitive
        values will be collapsed
    show_avals: bool
        If `True` type information will be included in the node label

    Returns
    -------
    Graph
        Pydot graph
    """

    g = pydot.Dot(graph_type="digraph")

    sub_graph, _, _, _ = graph.sub_graph(x.eqns[0], 0, collapse_primitives, show_avals)
    g.add_subgraph(sub_graph)

    return g
