import typing

import pydot
from jax._src import core as jax_core

from . import styling, utils


def get_arg_node(
    arg_id: str,
    var: typing.Union[jax_core.Var, jax_core.Literal],
    show_avals: bool,
    is_literal: bool,
    id_map: utils.IdMap,
) -> pydot.Node:
    """
    Return a pydot node representing a function input/argument

    Parameters
    ----------
    arg_id: str
        Unique ID of the node
    var: jax._src.core.Var
        JAX variable or literal instance
    show_avals: bool
        If `True` show the type in the node
    is_literal: True
        Should be `True` if the node is a literal (and
        should be styled as such)
    id_map: IdMap
        Node id to label map

    Returns
    -------
    pydot.Node
    """
    style = styling.LITERAL_STYLING if is_literal else styling.IN_ARG_STYLING
    return pydot.Node(
        name=arg_id,
        label=utils.get_node_label(var, show_avals, id_map),
        **style,
    )


def get_const_node(
    arg_id: str,
    var: typing.Union[jax_core.Var, jax_core.Literal],
    show_avals: bool,
    id_map: utils.IdMap,
) -> pydot.Node:
    """
    Return a pydot node representing a function const arg

    Parameters
    ----------
    arg_id: str
        Unique ID of the node
    var: jax._src.core.Var
        JAX variable
    show_avals: bool
        If `True` show the type in the node
    id_map: IdMap
        Node id to label map

    Returns
    -------
    pydot.Node
    """
    return pydot.Node(
        name=arg_id,
        label=utils.get_node_label(var, show_avals, id_map),
        **styling.CONST_ARG_STYLING,
    )


def get_var_node(
    var_id: str, var: jax_core.Var, show_avals: bool, id_map: utils.IdMap
) -> pydot.Node:
    """
    Get a pydot node representing a variable internal to a function

    Parameters
    ----------
    var_id: str
        Unique ID of the node
    var: jax._src.core.Var
        JAX variable instance
    show_avals: bool
        If `True` show the type in the node
    id_map: IdMap
        Node id to label map

    Returns
    -------
    pydot.Node
    """
    return pydot.Node(
        name=var_id,
        label=utils.get_node_label(var, show_avals, id_map),
        **styling.VAR_STYLING,
    )


def get_out_node(
    out_id: str, var: jax_core.Var, show_avals: bool, id_map: utils.IdMap
) -> pydot.Node:
    """
    Get a pydot node representing the outputs of a function

    Parameters
    ----------
    out_id: str
        Unique ID of the node
    var: jax._src.core.Var
        JAX variable instance
    show_avals: bool
        If `True` show the type in the node
    id_map: IdMap
        Node id to label map

    Returns
    -------
    pydot.Node
    """
    return pydot.Node(
        name=out_id,
        label=utils.get_node_label(var, show_avals, id_map),
        **styling.OUT_ARG_STYLING,
    )


def get_subgraph(graph_id: str, label: str) -> pydot.Subgraph:
    """
    Get a pydot subgraph

    Parameters
    ----------
    graph_id: str
        Unique ID of the subgraph
    label: str
        Label of the subgraph

    Returns
    -------
    pydot.Subgraph
    """
    return pydot.Subgraph(
        graph_id,
        label=label,
        rank="same",
        **styling.GRAPH_STYLING,
    )


def get_arguments(
    graph_id: str,
    parent_id: str,
    graph_consts: typing.List[jax_core.Var],
    graph_invars: typing.List[jax_core.Var],
    parent_invars: typing.List[jax_core.Var],
    show_avals: bool,
    id_map: utils.IdMap,
) -> typing.Tuple[pydot.Subgraph, typing.List[pydot.Edge]]:
    """
    Generate a subgraph containing arguments, and edges connecting
    it to its parent graph

    Parameters
    ----------
    graph_id: str
        ID of the subgraph that owns the arguments
    parent_id: str
        ID of the parent of the subgraph
    graph_consts: List[jax._src.core.Var]
        List of graph const-vars
    graph_invars: List[jax._src.core.Var]
        List of input variables to the subgraph
    parent_invars: List[jax._src.core.Var]
        List of the corresponding input variables from the parent subgraph
    show_avals: bool
        If `True` show the type in the node
    id_map: IdMap
        Node id to label map

    Returns
    -------
    (pydot.Subgraph, typing.List[pydot.Edge])
        Tuple containing the argument subgraph and a list of
        edges that connect variables in the parent graph to
        the inputs of this subgraph.
    """
    argument_nodes = pydot.Subgraph(
        f"{graph_id}_args", rank="same", **styling.ARG_SUBGRAPH_STYLING
    )
    argument_edges = list()

    for var in graph_consts:
        arg_id = f"{graph_id}_{id(var)}"
        argument_nodes.add_node(get_const_node(arg_id, var, show_avals, id_map))

    for var, p_var in zip(graph_invars, parent_invars):
        # TODO: What does the underscore mean?
        if str(var)[-1] == "_":
            continue
        arg_id = f"{graph_id}_{id(var)}"
        is_literal = isinstance(var, jax_core.Literal)
        argument_nodes.add_node(
            get_arg_node(arg_id, var, show_avals, is_literal, id_map)
        )
        if not is_literal:
            argument_edges.append(pydot.Edge(f"{parent_id}_{id(p_var)}", arg_id))

    return argument_nodes, argument_edges


def get_scan_arguments(
    graph_id: str,
    parent_id: str,
    graph_invars: typing.List[jax_core.Var],
    parent_invars: typing.List[jax_core.Var],
    n_const: int,
    n_carry: int,
    show_avals: bool,
    id_map: utils.IdMap,
) -> typing.Tuple[pydot.Subgraph, typing.List[pydot.Edge]]:
    """
    Generate a subgraph containing arguments, and edges connecting
    it to its parent graph. Groups scan init/carry nodes.

    Parameters
    ----------
    graph_id: str
        ID of the subgraph that owns the arguments
    parent_id: str
        ID of the parent of the subgraph
    graph_invars: List[jax._src.core.Var]
        List of input variables to the subgraph
    parent_invars: List[jax._src.core.Var]
        List of the corresponding input variables from the parent subgraph
    n_const: int
        Number of scan constant arguments
    n_carry: int
        Number of scan carry arguments
    show_avals: bool
        If `True` show the type in the node
    id_map: IdMap
        Node id to label map

    Returns
    -------
    (pydot.Subgraph, typing.List[pydot.Edge])
        Tuple containing the argument subgraph and a list of
        edges that connect variables in the parent graph to
        the inputs of this subgraph.
    """
    argument_nodes = pydot.Subgraph(
        f"cluster_{graph_id}_args", rank="min", **styling.ARG_SUBGRAPH_STYLING
    )
    const_nodes = pydot.Subgraph(
        f"cluster_{graph_id}_const", rank="same", label="init", style="dotted"
    )
    carry_nodes = pydot.Subgraph(
        f"cluster_{graph_id}_init", rank="same", label="consts", style="dotted"
    )
    iterate_nodes = pydot.Subgraph(
        f"cluster_{graph_id}_iter", rank="same", label="iterate", style="dotted"
    )
    argument_edges = list()

    for i, (var, p_var) in enumerate(zip(graph_invars, parent_invars)):
        # TODO: What does the underscore mean?
        if str(var)[-1] == "_":
            continue
        arg_id = f"{graph_id}_{id(var)}"

        var_is_literal = isinstance(var, jax_core.Literal)
        parent_is_literal = isinstance(p_var, jax_core.Literal)
        is_literal = var_is_literal or parent_is_literal

        if parent_is_literal:
            literal_id = f"{arg_id}_lit"
            argument_nodes.add_node(
                get_arg_node(literal_id, p_var, show_avals, True, id_map)
            )
            argument_nodes.add_edge(pydot.Edge(literal_id, arg_id))

        if i < n_const:
            const_nodes.add_node(
                get_arg_node(arg_id, var, show_avals, var_is_literal, id_map)
            )
        elif i < n_carry + n_const:
            carry_nodes.add_node(
                get_arg_node(arg_id, var, show_avals, var_is_literal, id_map)
            )
        else:
            iterate_nodes.add_node(
                get_arg_node(arg_id, var, show_avals, var_is_literal, id_map)
            )

        if not is_literal:
            argument_edges.append(pydot.Edge(f"{parent_id}_{id(p_var)}", arg_id))

    argument_nodes.add_subgraph(const_nodes)
    argument_nodes.add_subgraph(carry_nodes)
    argument_nodes.add_subgraph(iterate_nodes)
    return argument_nodes, argument_edges


def get_outputs(
    graph_id: str,
    parent_id: str,
    graph_invars: typing.List[jax_core.Var],
    graph_outvars: typing.List[jax_core.Var],
    parent_outvars: typing.List[jax_core.Var],
    show_avals: bool,
    id_map: utils.IdMap,
) -> typing.Tuple[
    pydot.Subgraph,
    typing.List[pydot.Edge],
    typing.List[pydot.Node],
    typing.List[pydot.Edge],
]:
    """
    Generate a subgraph containing function output nodes, and
    edges and nodes that connect outputs to the parent graph

    Parameters
    ----------
    graph_id: str
        ID of the subgraph
    parent_id: str
        ID of the parent graph
    graph_invars: List[jax._src.core.Var]
        List of funtion input variables
    graph_outvars: List[jax._src.core.Var]
        List of output function variables
    parent_outvars: List[jax._src.core.Var]
        Corresponding list of variable from the parent
        graph that are outputs from this graph
    show_avals: bool
        If `True` show the type in the node
    id_map: IdMap
        Node id to label map

    Returns
    -------
    (
        pydot.Subgraph,
        typing.List[pydot.Edge],
        typing.List[pydot.Node],
        typing.List[pydot.Edge]
    )
        Tuple containing:
            - The subgraph wrapping the output nodes
            - A list of edges connecting to the parent graph
            - A list of variable nodes that should be added to the
              parent graph (as outputs from this graph)
            - A list of edges that connect inputs directly to outputs
              in the case an argument is returned by the function
    """
    out_graph = pydot.Subgraph(
        f"{graph_id}_outs", rank="same", **styling.ARG_SUBGRAPH_STYLING
    )
    out_edges = list()
    out_nodes = list()
    id_edges = list()
    in_var_set = set([str(x) for x in graph_invars])

    for var, p_var in zip(graph_outvars, parent_outvars):
        if str(var) in in_var_set:
            arg_id = f"{graph_id}_{id(var)}_out"
            id_edges.append(pydot.Edge(f"{graph_id}_{id(var)}", arg_id))
        else:
            arg_id = f"{graph_id}_{id(var)}"
        out_graph.add_node(get_out_node(arg_id, var, show_avals, id_map))
        out_edges.append(pydot.Edge(arg_id, f"{parent_id}_{id(p_var)}"))
        out_nodes.append(
            get_var_node(f"{parent_id}_{id(p_var)}", p_var, show_avals, id_map)
        )

    return out_graph, out_edges, out_nodes, id_edges


def get_scan_outputs(
    graph_id: str,
    parent_id: str,
    graph_invars: typing.List[jax_core.Var],
    graph_outvars: typing.List[jax_core.Var],
    parent_outvars: typing.List[jax_core.Var],
    n_carry: int,
    show_avals: bool,
    id_map: utils.IdMap,
) -> typing.Tuple[
    pydot.Subgraph,
    typing.List[pydot.Edge],
    typing.List[pydot.Node],
    typing.List[pydot.Edge],
]:
    """
    Generate a subgraph containing function output nodes, and
    edges and nodes that connect outputs to the parent graph.
    Groups carry nodes.

    Parameters
    ----------
    graph_id: str
        ID of the subgraph
    parent_id: str
        ID of the parent graph
    graph_invars: List[jax._src.core.Var]
        List of funtion input variables
    graph_outvars: List[jax._src.core.Var]
        List of output function variables
    parent_outvars: List[jax._src.core.Var]
        Corresponding list of variable from the parent
        graph that are outputs from this graph
    n_carry: int
        Number of scan carry arguments
    show_avals: bool
        If `True` show the type in the node
    id_map: IdMap
        Node id to label map

    Returns
    -------
    (
        pydot.Subgraph,
        typing.List[pydot.Edge],
        typing.List[pydot.Node],
        typing.List[pydot.Edge]
    )
        Tuple containing:
            - The subgraph wrapping the output nodes
            - A list of edges connecting to the parent graph
            - A list of variable nodes that should be added to the
              parent graph (as outputs from this graph)
            - A list of edges that connect inputs directly to outputs
              in the case an argument is returned by the function
    """
    out_graph = pydot.Subgraph(
        f"cluster_{graph_id}_outs", rank="same", **styling.ARG_SUBGRAPH_STYLING
    )
    carry_nodes = pydot.Subgraph(
        f"cluster_{graph_id}_carry", rank="same", label="carry", style="dotted"
    )
    accumulate_nodes = pydot.Subgraph(
        f"cluster_{graph_id}_acc", rank="same", label="accumulate", style="dotted"
    )
    out_edges = list()
    out_nodes = list()
    id_edges = list()
    in_var_set = set([id(x) for x in graph_invars])

    for i, (var, p_var) in enumerate(zip(graph_outvars, parent_outvars)):
        if id(var) in in_var_set:
            arg_id = f"{graph_id}_{id(var)}_out"
            id_edges.append(pydot.Edge(f"{graph_id}_{id(var)}", arg_id))
        else:
            arg_id = f"{graph_id}_{id(var)}"
        if i < n_carry:
            carry_nodes.add_node(get_out_node(arg_id, var, show_avals, id_map))
        else:
            accumulate_nodes.add_node(get_out_node(arg_id, var, show_avals, id_map))
        out_edges.append(pydot.Edge(arg_id, f"{parent_id}_{id(p_var)}"))
        out_nodes.append(
            get_var_node(f"{parent_id}_{id(p_var)}", p_var, show_avals, id_map)
        )

    out_graph.add_subgraph(carry_nodes)
    out_graph.add_subgraph(accumulate_nodes)
    return out_graph, out_edges, out_nodes, id_edges
