import typing

import jax._src.core as jax_core
import pydot

GRAPH_STYLING = dict(
    fontname="Courier",
    fontsize="12",
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
COND_NODE_STYLING = dict(
    shape="box",
    fontname="Courier",
    fontsize="10",
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


def _add_arguments(
    graph: pydot.Subgraph,
    graph_id: str,
    parent_graph: pydot.Subgraph,
    parent_id: str,
    vars: typing.List[jax_core.Var],
    show_avals: bool,
):
    arg_subgraph = pydot.Subgraph(f"{graph_id}_args", rank="same")
    for var in vars:
        arg_id = f"{graph_id}_{var}"
        styling = (
            LITERAL_STYLING if isinstance(var, jax_core.Literal) else IN_ARG_STYLING
        )
        arg_subgraph.add_node(
            pydot.Node(
                name=arg_id,
                label=_get_node_label(var, show_avals),
                **styling,
            )
        )
        if not isinstance(var, jax_core.Literal):
            parent_graph.add_edge(pydot.Edge(f"{parent_id}_{var}", arg_id))
    graph.add_subgraph(arg_subgraph)


def _add_outputs(
    graph: pydot.Subgraph,
    graph_id: str,
    parent_graph: pydot.Subgraph,
    parent_id: str,
    vars: typing.List[jax_core.Var],
    show_avals: bool,
):
    for var in vars:
        arg_id = f"{graph_id}_{var}"
        graph.add_node(
            pydot.Node(
                name=arg_id,
                label=_get_node_label(var, show_avals),
                **OUT_ARG_STYLING,
            )
        )
        parent_graph.add_edge(pydot.Edge(arg_id, f"{parent_id}_{var}"))


def _add_conditional(
    conditional,
    graph: pydot.Subgraph,
    graph_id: str,
    collapse_primitives: bool,
    show_avals: bool,
    n: int,
) -> typing.Tuple[pydot.Subgraph, int]:
    cond_graph = pydot.Subgraph(
        f"cluster_{graph_id}_cond_{n}",
        rank="same",
        label="cond",
        **GRAPH_STYLING,
    )
    n = n + 1

    cond_node_id = f"{graph_id}_cond_{n}"
    cond_graph.add_node(
        pydot.Node(name=cond_node_id, label="cond", **COND_NODE_STYLING)
    )

    cond_var = conditional.invars[0]

    if isinstance(cond_var, jax_core.Literal):
        graph.add_node(
            pydot.Node(
                name=f"{graph_id}_{cond_var}",
                label=_get_node_label(cond_var, show_avals),
                **LITERAL_STYLING,
            )
        )
    graph.add_edge(pydot.Edge(f"{graph_id}_{cond_var}", cond_node_id))
    _add_arguments(
        cond_graph, cond_node_id, graph, graph_id, conditional.invars[1:], show_avals
    )
    _add_outputs(
        cond_graph, cond_node_id, graph, graph_id, conditional.outvars, show_avals
    )
    graph.add_subgraph(cond_graph)

    for i, branch in enumerate(conditional.params["branches"]):

        if len(branch.eqns) == 0:
            branch_graph_id = f"{cond_node_id}_branch_{i}"
            branch_graph = pydot.Subgraph(
                f"cluster_{branch_graph_id}",
                label=f"Branch {i}",
                rank="same",
                **GRAPH_STYLING,
            )
            for var, c_var in zip(branch.jaxpr.outvars, conditional.outvars):
                arg_id = f"{branch_graph_id}_{var}"
                branch_graph.add_node(
                    pydot.Node(
                        name=arg_id,
                        label=_get_node_label(var, show_avals),
                        **VAR_STYLING,
                    )
                )
                cond_graph.add_edge(pydot.Edge(f"{cond_node_id}_{var}", arg_id))
                cond_graph.add_edge(pydot.Edge(arg_id, f"{cond_node_id}_{c_var}"))
            cond_graph.add_subgraph(branch_graph)
        elif len(branch.eqns[0].params) > 0:
            sub_graph, in_edges, out_edges, n = _sub_graph(
                branch.eqns[0], n, collapse_primitives, show_avals
            )
            sub_graph.set_label(f"Branch {i}: {sub_graph.get_label()}")
            cond_graph.add_subgraph(sub_graph)
            for (a, b) in in_edges:
                cond_graph.add_edge(pydot.Edge(f"{cond_node_id}_{a}", b))
            for ((a, b, c), d) in zip(out_edges, conditional.outvars):
                cond_graph.add_edge(pydot.Edge(b, f"{cond_node_id}_{d}"))
        else:
            branch_graph_id = f"{cond_node_id}_branch_{i}"
            branch_graph = pydot.Subgraph(
                f"cluster_{branch_graph_id}",
                label=f"Branch {i}",
                rank="same",
                **GRAPH_STYLING,
            )
            _add_arguments(
                branch_graph,
                branch_graph_id,
                cond_graph,
                cond_node_id,
                branch.eqns[0].invars,
                show_avals,
            )
            branch_graph, n = _add_node(
                branch.eqns[0], branch_graph, branch_graph_id, show_avals, n, True
            )
            for var, c_var in zip(branch.eqns[0].outvars, conditional.outvars):
                arg_id = f"{branch_graph_id}_{var}"
                branch_graph.add_node(
                    pydot.Node(
                        name=arg_id,
                        label=_get_node_label(var, show_avals),
                        **OUT_ARG_STYLING,
                    )
                )
                cond_graph.add_edge(pydot.Edge(arg_id, f"{cond_node_id}_{c_var}"))
            cond_graph.add_subgraph(branch_graph)

        n = n + 1

    return graph, n


def _get_arguments(
    x: jax_core.JaxprEqn,
    graph_id: str,
    show_avals: bool,
) -> typing.Tuple[pydot.Subgraph, typing.List]:
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

    argument_nodes = pydot.Subgraph(f"{graph_id}_args", rank="same")
    argument_edges = list()

    for (va, vb) in zip(x.invars, x.params["jaxpr"].jaxpr.invars):
        argument_nodes.add_node(
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
) -> typing.Tuple[pydot.Subgraph, typing.List]:
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

    output_nodes = pydot.Subgraph(f"{graph_id}_outputs")
    output_edges = list()

    for (va, vb) in zip(x.outvars, x.params["jaxpr"].jaxpr.outvars):
        output_nodes.add_node(
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
    graph.add_subgraph(argument_nodes)

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
            if eqn.primitive.name == "cond":
                graph, n = _add_conditional(
                    eqn,
                    graph,
                    graph_id,
                    collapse_primitives,
                    show_avals,
                    n,
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
    graph.add_subgraph(output_nodes)

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
