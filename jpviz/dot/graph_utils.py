import typing

import pydot
from jax._src import core as jax_core

from . import styling, utils


def get_arg_node(
    arg_id: str,
    var: typing.Union[jax_core.Var, jax_core.Literal],
    show_avals: bool,
    is_literal: bool,
) -> pydot.Node:
    style = styling.LITERAL_STYLING if is_literal else styling.IN_ARG_STYLING
    return pydot.Node(
        name=arg_id,
        label=utils.get_node_label(var, show_avals),
        **style,
    )


def get_var_node(var_id: str, var: jax_core.Var, show_avals: bool) -> pydot.Node:
    return pydot.Node(
        name=var_id,
        label=utils.get_node_label(var, show_avals),
        **styling.VAR_STYLING,
    )


def get_out_node(out_id: str, var: jax_core.Var, show_avals: bool) -> pydot.Node:
    return pydot.Node(
        name=out_id,
        label=utils.get_node_label(var, show_avals),
        **styling.OUT_ARG_STYLING,
    )


def get_subgraph(graph_id: str, label: str) -> pydot.Subgraph:
    return pydot.Subgraph(
        graph_id,
        label=label,
        rank="same",
        **styling.GRAPH_STYLING,
    )


def get_arguments(
    graph_id: str,
    parent_id: str,
    graph_invars: typing.List[jax_core.Var],
    parent_invars: typing.List[jax_core.Var],
    show_avals: bool,
) -> typing.Tuple[pydot.Subgraph, typing.List[pydot.Edge]]:
    argument_nodes = pydot.Subgraph(f"{graph_id}_args", rank="same")
    argument_edges = list()

    for var, p_var in zip(graph_invars, parent_invars):
        # TODO: What does the underscore mean?
        if str(var)[-1] == "_":
            continue
        arg_id = f"{graph_id}_{var}"
        is_literal = isinstance(var, jax_core.Literal)
        argument_nodes.add_node(get_arg_node(arg_id, var, show_avals, is_literal))
        if not is_literal:
            argument_edges.append(pydot.Edge(f"{parent_id}_{p_var}", arg_id))

    return argument_nodes, argument_edges


def get_outputs(
    graph_id: str,
    parent_id: str,
    graph_invars: typing.List[jax_core.Var],
    graph_outvars: typing.List[jax_core.Var],
    parent_outvars: typing.List[jax_core.Var],
    show_avals: bool,
) -> typing.Tuple[
    pydot.Subgraph,
    typing.List[pydot.Edge],
    typing.List[pydot.Node],
    typing.List[pydot.Edge],
]:
    out_graph = pydot.Subgraph(f"{graph_id}_outs", rank="same")
    out_edges = list()
    out_nodes = list()
    id_edges = list()
    in_var_set = set([str(x) for x in graph_invars])

    for var, p_var in zip(graph_outvars, parent_outvars):
        if str(var) in in_var_set:
            arg_id = f"{graph_id}_{var}_out"
            id_edges.append(pydot.Edge(f"{graph_id}_{var}", arg_id))
        else:
            arg_id = f"{graph_id}_{var}"
        out_graph.add_node(get_out_node(arg_id, var, show_avals))
        out_edges.append(pydot.Edge(arg_id, f"{parent_id}_{p_var}"))
        out_nodes.append(get_var_node(f"{parent_id}_{p_var}", p_var, show_avals))

    return out_graph, out_edges, out_nodes, id_edges
