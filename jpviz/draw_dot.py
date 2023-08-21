import typing

import jax._src.core as jax_core
import pydot

GRAPH_STYLING = dict(
    fontname="Courier",
    fontsize="15",
    style="dotted",
    labeljust="l",
)
IN_ARG_STYLING = dict(
    shape="box",
    color="green",
    fontname="Courier",
    fontsize="10",
)
LITERAL_STYLING = dict(
    shape="box",
    color="orange",
    fontname="Courier",
    fontsize="10",
)
OUT_ARG_STYLING = dict(
    shape="box",
    color="red",
    fontname="Courier",
    fontsize="10",
)
VAR_STYLING = dict(
    shape="box",
    color="blue",
    fontname="Courier",
    fontsize="10",
)
PRIMITIVE_STYLING = dict(
    shape="oval",
    fontname="Courier",
    fontsize="10",
    linestye="dashed",
)
FUNCTION_NODE_STYLING = dict(
    shape="rectangle", fontname="Courier", fontsize="10", style="dotted"
)


def _get_node_label(
    v: typing.Union[jax_core.Var, jax_core.Literal], show_avals: bool
) -> str:
    """
    Concatenate a variable name and its type.

    Parameters
    ----------
    v: Var
        Jax variable
    show_avals: bool
        If `True` then the type will be included in the
        node label

    Returns
    -------
    str
    """
    if show_avals:
        return f"{v}: {v.aval.str_short()}"
    else:
        return str(v)


def _is_not_primitive(x: jax_core.JaxprEqn) -> bool:
    """
    Test if a JaxprEqn is a primitive.

    Parameters
    ----------
    x: JaxprEqn

    Returns
    -------
    bool
        'True' if not a primitive.
    """
    return "jaxpr" in x.params


def _contains_non_primitives(x: jax_core.JaxprEqn) -> bool:
    """
    Check if a JaxprEqn itself contains any non-primitive nodes.

    Parameters
    ----------
    x : JaxPr

    Returns
    -------
    bool
        `True` if any sub nodes are non-primitive
    """
    eqns = [e for e in x.params["jaxpr"].jaxpr.eqns]
    return any(["jaxpr" in e.params for e in eqns])


def _add_node(
    x: jax_core.JaxprEqn,
    graph: pydot.Subgraph,
    graph_id: str,
    show_avals: bool,
    n: int,
    is_primitive: bool,
) -> typing.Tuple[pydot.Subgraph, int]:
    """
    Add a node representing a function to the graph.

    Parameters
    ----------
    x: JaxprEqn
    graph: pydot.Subgraph
        Subgraph to add nodes to
    graph_id: str
        Graph identifier
    show_avals: bool
        If `True` variable types will be included in labels
    n: int
        Used to ensure unique node ids
    is_primitive: bool
        If 'True' node will be styled as primitive (i.e. built-in) function

    Returns
    -------
    (pydot.Subgraph, int)
        Updated graph and counter
    """
    name = str(x.primitive) if is_primitive else x.params["name"]
    node_id = f"{graph_id}_{name}_{n}"

    styling = PRIMITIVE_STYLING if is_primitive else FUNCTION_NODE_STYLING

    graph.add_node(pydot.Node(name=node_id, label=name, **styling))

    for var in x.invars:
        if isinstance(var, jax_core.Literal):
            graph.add_node(
                pydot.Node(
                    name=f"{graph_id}_{var}",
                    label=_get_node_label(var, show_avals),
                    **LITERAL_STYLING,
                )
            )
        graph.add_edge(pydot.Edge(f"{graph_id}_{var}", node_id))

    for var in x.outvars:
        graph.add_node(
            pydot.Node(
                name=f"{graph_id}_{var}",
                label=_get_node_label(var, show_avals),
                **VAR_STYLING,
            )
        )
        graph.add_edge(pydot.Edge(node_id, f"{graph_id}_{var}"))

    return graph, n + 1


def _get_arguments(
    x: jax_core.JaxprEqn,
    graph_id: str,
    show_avals: bool,
) -> typing.Tuple[typing.List, typing.List]:
    """
    Get a list of argument nodes and outer edges

    Parameters
    ----------
    x: JaxprEqn
    graph_id: str
        Id of the function graph
    show_avals: bool
        If `True` types will be included in the node label

    Returns
    -------
    (list, list)
        List of nodes to be added to the graph, and list
        of external edges linking to the parent graph
    """

    argument_nodes = list()
    argument_edges = list()

    for (va, vb) in zip(x.invars, x.params["jaxpr"].jaxpr.invars):
        argument_nodes.append(
            pydot.Node(
                name=f"{graph_id}_{vb}",
                label=_get_node_label(vb, show_avals),
                **IN_ARG_STYLING,
            )
        )
        argument_edges.append((va, f"{graph_id}_{vb}"))

    return argument_nodes, argument_edges


def _get_outputs(
    x: jax_core.JaxprEqn,
    graph_id: str,
    show_avals: bool,
) -> typing.Tuple[typing.List, typing.List]:
    """
    Get a list of output nodes and outer edges

    Parameters
    ----------
    x: JaxprEqn
    graph_id: str
        Id of the function graph
    show_avals: bool
        If `True` types will be included in the node label

    Returns
    -------
    (list, list)
        List of nodes to be added to the graph, and list
        of external edges linking to the parent graph
    """

    output_nodes = list()
    output_edges = list()

    for (va, vb) in zip(x.outvars, x.params["jaxpr"].jaxpr.outvars):
        output_nodes.append(
            pydot.Node(
                name=f"{graph_id}_{vb}",
                label=_get_node_label(vb, show_avals),
                **OUT_ARG_STYLING,
            )
        )
        output_edges.append((va, f"{graph_id}_{vb}", va))

    return output_nodes, output_edges


def _sub_graph(
    x: jax_core.JaxprEqn,
    n: int,
    collapse_primitives: bool,
    show_avals: bool,
) -> typing.Tuple[pydot.Subgraph, typing.List, typing.List, int]:
    """
    Create a subgraph representing a jitted function

    Parameters
    ----------
    x: JaxprEqn
    n: int
        Counter used to ensure unique node ids
    collapse_primitives: bool
        If `True` nodes that are composed of only
        primitives will be collapses to a single node
    show_avals: bool
        If `True` type information will be included
        on variable nodes

    Returns
    -------
    (SubGraph, list, list, int)
        Tuple containing:
            - The pydot subgraph
            - List of argument edges to link to the parent graph
            - List of output edges to link to the parent graph
            - Incremented counter
    """

    graph_name = x.params["name"]
    graph_id = f"{graph_name}_{n}"
    n = n + 1

    graph = pydot.Subgraph(
        f"cluster_{graph_id}", rank="same", label=graph_name, **GRAPH_STYLING
    )

    argument_nodes, argument_edges = _get_arguments(x, graph_id, show_avals)

    for node in argument_nodes:
        graph.add_node(node)

    eqns = [z for z in x.params["jaxpr"].jaxpr.eqns]

    for eqn in eqns:
        if _is_not_primitive(eqn):
            if _contains_non_primitives(eqn) or not collapse_primitives:
                sub_graph, in_edges, out_edges, n = _sub_graph(
                    eqn, n, collapse_primitives, show_avals
                )
                graph.add_subgraph(sub_graph)
                for (a, b) in in_edges:
                    graph.add_edge(pydot.Edge(f"{graph_id}_{a}", b))
                for (a, b, c) in out_edges:
                    graph.add_node(
                        pydot.Node(
                            name=f"{graph_id}_{a}",
                            label=_get_node_label(a, show_avals),
                            **VAR_STYLING,
                        )
                    )
                    graph.add_edge(pydot.Edge(b, f"{graph_id}_{c}"))
            else:
                graph, n = _add_node(
                    eqn,
                    graph,
                    graph_id,
                    show_avals,
                    n,
                    False,
                )
        else:
            graph, n = _add_node(
                eqn,
                graph,
                graph_id,
                show_avals,
                n,
                True,
            )

    output_nodes, output_edges = _get_outputs(x, graph_id, show_avals)

    for node in output_nodes:
        graph.add_node(node)

    return graph, argument_edges, output_edges, n


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

    sub_graph, _, _, _ = _sub_graph(x.eqns[0], 0, collapse_primitives, show_avals)
    g.add_subgraph(sub_graph)

    return g
