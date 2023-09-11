# flake8: noqa: C901
import typing

import jax._src.core as jax_core
import pydot

from . import graph_utils, styling, utils

sub_graph_return = typing.Tuple[
    typing.Union[pydot.Node, pydot.Subgraph],
    typing.List[pydot.Edge],
    typing.List[pydot.Node],
    typing.List[pydot.Edge],
    int,
]


def get_conditional(
    conditional: jax_core.Jaxpr,
    parent_id: str,
    n: int,
    collapse_primitives: bool,
    show_avals: bool,
) -> sub_graph_return:
    """
    Generate a subgraph representing a conditional function

    Parameters
    ----------
    conditional: jax._src.core.Jaxpr
        Jaxpr of the conditional function
    parent_id: str
        ID of the parent subgraph of the conditional node
    collapse_primitives: bool
        If `True` any subgraph only consisting of primitive
        functions is collapsed into a single node
    show_avals: bool
        If `True` the type of the data is shown on
        argument/variable nodes on the generated graph
    n: int
        Integer used to generate unique ids for nodes, incremented
        as new nodes are added

    Returns
    -------
    (
        typing.Union[pydot.Node, pydot.Subgraph],
        typing.List[pydot.Edge],
        typing.List[pydot.Node],
        typing.List[pydot.Edge],
        int
    )
        Tuple containing:
            - Subgraph representing the conditional function and branches
            - List of edges that will connect a parent graph to the
              arguments of the conditional function
            - List of nodes that should be added to a parent graph (i.e.
              outputs of this graph)
            - List of edges connecting the outputs of this graph to
              parent graph
            - Updated incremented integer used to get unique node ids
    """

    cond_graph_id = f"{parent_id}_cond_{n}"
    cond_graph = graph_utils.get_subgraph(f"cluster_{cond_graph_id}", "switch")
    n = n + 1

    cond_node_id = f"{cond_graph_id}_node"
    cond_arguments = pydot.Subgraph(f"{cond_graph_id}_inputs", rank="same")
    cond_arguments.add_node(
        pydot.Node(name=cond_node_id, label="idx", **styling.COND_NODE_STYLING)
    )

    in_edges = list()
    new_nodes = list()
    out_edges = list()

    cond_var = conditional.invars[0]
    cond_var_id = f"{parent_id}_{cond_var}"
    if isinstance(cond_var, jax_core.Literal):
        new_nodes.append(
            graph_utils.get_arg_node(cond_var_id, cond_var, show_avals, True)
        )
    in_edges.append(pydot.Edge(cond_var_id, cond_node_id))

    for arg in conditional.invars[1:]:
        arg_id = f"{cond_graph_id}_{arg}"
        is_literal = isinstance(arg, jax_core.Literal)
        cond_arguments.add_node(
            graph_utils.get_arg_node(arg_id, arg, show_avals, is_literal)
        )
        in_edges.append(pydot.Edge(f"{parent_id}_{arg}", arg_id))

    cond_graph.add_subgraph(cond_arguments)

    for i, branch in enumerate(conditional.params["branches"]):
        if len(branch.eqns) == 0:
            branch_graph_id = f"{cond_node_id}_branch_{i}"
            label = f"Branch {i}"

            if collapse_primitives:
                cond_graph.add_node(
                    pydot.Node(
                        name=branch_graph_id,
                        label=label,
                        **styling.FUNCTION_NODE_STYLING,
                    )
                )

                for (var, p_var) in zip(branch.jaxpr.invars, conditional.invars[1:]):
                    # TODO: What does the underscore mean?
                    if str(var)[-1] == "_":
                        continue
                    cond_graph.add_edge(
                        pydot.Edge(f"{cond_graph_id}_{p_var}", branch_graph_id)
                    )
                for var in conditional.outvars:
                    cond_graph.add_edge(
                        pydot.Edge(branch_graph_id, f"{cond_graph_id}_{var}")
                    )
            else:
                branch_graph = graph_utils.get_subgraph(
                    f"cluster_{branch_graph_id}", label
                )
                for (var, p_var) in zip(branch.jaxpr.invars, conditional.invars[1:]):
                    # TODO: What does the underscore mean?
                    if str(var)[-1] == "_":
                        continue
                    arg_id = f"{branch_graph_id}_{var}"
                    branch_graph.add_node(
                        graph_utils.get_var_node(arg_id, var, show_avals)
                    )
                    cond_graph.add_edge(pydot.Edge(f"{cond_graph_id}_{p_var}", arg_id))
                for var, c_var in zip(branch.jaxpr.outvars, conditional.outvars):
                    arg_id = f"{branch_graph_id}_{var}"
                    cond_graph.add_edge(pydot.Edge(arg_id, f"{cond_graph_id}_{c_var}"))
                cond_graph.add_subgraph(branch_graph)
        else:
            branch_graph_id = f"{cond_node_id}_branch_{i}"
            if len(branch.eqns) == 1:
                eqn = branch.eqns[0]
                branch_label = (
                    eqn.params["name"] if "name" in eqn.params else eqn.primitive.name
                )
                branch_label = f"Branch {i}: {branch_label}"
                no_literal_inputs = any(
                    [isinstance(a, jax_core.Literal) for a in branch.jaxpr.invars]
                )
                collapse_branch = no_literal_inputs or collapse_primitives
            else:
                branch_label = f"Branch {i}"
                collapse_branch = collapse_primitives

            if utils.contains_non_primitives(branch.eqns) or not collapse_branch:
                branch_graph = graph_utils.get_subgraph(
                    f"cluster_{branch_graph_id}", branch_label
                )
                branch_args, arg_edges = graph_utils.get_arguments(
                    branch_graph_id,
                    cond_graph_id,
                    branch.jaxpr.constvars,
                    branch.jaxpr.invars,
                    conditional.invars[1:],
                    show_avals,
                )
                for edge in arg_edges:
                    cond_graph.add_edge(edge)
                branch_graph.add_subgraph(branch_args)

                for eqn in branch.eqns:
                    (
                        eqn_graph,
                        eqn_in_edges,
                        eqn_out_nodes,
                        eqn_out_edges,
                        n,
                    ) = get_sub_graph(
                        eqn,
                        branch_graph_id,
                        n,
                        collapse_primitives,
                        show_avals,
                    )
                    if isinstance(eqn_graph, pydot.Subgraph):
                        branch_graph.add_subgraph(eqn_graph)
                    else:
                        branch_graph.add_node(eqn_graph)
                    for edge in eqn_in_edges:
                        branch_graph.add_edge(edge)
                    for node in eqn_out_nodes:
                        branch_graph.add_node(node)
                    for edge in eqn_out_edges:
                        branch_graph.add_edge(edge)

                (
                    branch_out_graph,
                    branch_out_edges,
                    branch_out_nodes,
                    id_edges,
                ) = graph_utils.get_outputs(
                    branch_graph_id,
                    cond_graph_id,
                    branch.jaxpr.invars,
                    branch.jaxpr.outvars,
                    conditional.outvars,
                    show_avals,
                )
                branch_graph.add_subgraph(branch_out_graph)
                for edge in branch_out_edges:
                    cond_graph.add_edge(edge)
                for node in branch_out_nodes:
                    cond_graph.add_node(node)
                for edge in id_edges:
                    branch_graph.add_edge(edge)

                cond_graph.add_subgraph(branch_graph)
            else:
                cond_graph.add_node(
                    pydot.Node(
                        name=branch_graph_id,
                        label=branch_label,
                        **styling.FUNCTION_NODE_STYLING,
                    )
                )
                for (var, p_var) in zip(branch.jaxpr.invars, conditional.invars[1:]):
                    # TODO: What does the underscore mean?

                    if str(var)[-1] == "_":
                        continue

                    if not is_literal:
                        cond_graph.add_edge(
                            pydot.Edge(f"{cond_graph_id}_{p_var}", branch_graph_id)
                        )

                for var in conditional.outvars:
                    cond_graph.add_edge(
                        pydot.Edge(branch_graph_id, f"{cond_graph_id}_{var}")
                    )

    cond_out_graph, cond_out_edges, cond_out_nodes, _ = graph_utils.get_outputs(
        cond_graph_id,
        parent_id,
        conditional.invars,
        conditional.outvars,
        conditional.outvars,
        show_avals,
    )
    cond_graph.add_subgraph(cond_out_graph)
    out_edges.extend(cond_out_edges)
    new_nodes.extend(cond_out_nodes)

    return cond_graph, in_edges, new_nodes, out_edges, n


def _get_node(
    eqn: jax_core.JaxprEqn,
    graph_id: str,
    show_avals: bool,
    n: int,
    is_primitive: bool,
) -> sub_graph_return:
    """
    Generate a node representing a function and edges connecting it
    to a parent graph

    Parameters
    ----------
    eqn: jax._src.core.JaxprEqn
        JaxprEqn of the function
    graph_id: str
        Id of the computation graph containing this node
    show_avals: bool
        If `True` the type of the data is shown on
        argument/variable nodes on the generated graph
    n: int
        Integer used to generate unique ids for nodes, incremented
        as new nodes are added
    is_primitive: bool
        Should be `True` if the function is a JAX primitive

    Returns
    -------
    (
        typing.Union[pydot.Node, pydot.Subgraph],
        typing.List[pydot.Edge],
        typing.List[pydot.Node],
        typing.List[pydot.Edge],
        int
    )
        Tuple containing:
            - Node representing the function
            - List of edges that will connect a parent graph to the
              arguments of the function
            - List of nodes that should be added to a parent graph (i.e.
              outputs of this graph)
            - List of edges connecting the outputs of this node to
              parent graph
            - Updated incremented integer used to get unique node ids
    """

    name = str(eqn.primitive) if is_primitive else eqn.params["name"]
    node_id = f"{name}_{n}"
    n = n + 1

    style = styling.PRIMITIVE_STYLING if is_primitive else styling.FUNCTION_NODE_STYLING
    node = pydot.Node(name=node_id, label=name, **style)

    new_nodes = list()
    in_edges = list()
    out_edges = list()

    for var in eqn.invars:
        if isinstance(var, jax_core.Literal):
            new_nodes.append(
                graph_utils.get_arg_node(f"{graph_id}_{var}", var, show_avals, True)
            )
        in_edges.append(pydot.Edge(f"{graph_id}_{var}", node_id))

    for var in eqn.outvars:
        var_id = f"{graph_id}_{var}"
        new_nodes.append(graph_utils.get_var_node(var_id, var, show_avals))
        out_edges.append(pydot.Edge(node_id, var_id))

    return node, in_edges, new_nodes, out_edges, n


def expand_non_primitive(
    eqn: jax_core.JaxprEqn,
    parent_id: str,
    n: int,
    collapse_primitives: bool,
    show_avals: bool,
    is_scan: bool = False,
) -> sub_graph_return:
    """
    Expand a JaxprEqn into a computation graph/

    Parameters
    ----------
    eqn: jax._src.core.JaxprEqn
        JaxprEqn of the function
    parent_id: str
        ID of the parent graph to this eqn
    n: int
        Integer used to generate unique ids for nodes, incremented
        as new nodes are added
    collapse_primitives: bool
        If `True` any functions that consist of only primitive
        elements will be collapsed to a single node
    show_avals: bool
        If `True` the type of the data is shown on
        argument/variable nodes on the generated graph
    is_scan: bool
        Should be `True` if the function primitive is 'scan'

    Returns
    -------
    (
        typing.Union[pydot.Node, pydot.Subgraph],
        typing.List[pydot.Edge],
        typing.List[pydot.Node],
        typing.List[pydot.Edge],
        int
    )
        Tuple containing:
            - Node representing the function
            - List of edges that will connect a parent graph to the
              arguments of the function
            - List of nodes that should be added to a parent graph (i.e.
              outputs of this graph)
            - List of edges connecting the outputs of this node to
              parent graph
            - Updated incremented integer used to get unique node ids
    """
    graph_name = eqn.params["name"] if "name" in eqn.params else eqn.primitive.name
    graph_id = f"{graph_name}_{n}"
    n = n + 1

    graph = pydot.Subgraph(
        f"cluster_{graph_id}",
        rank="same",
        label=graph_name,
        **styling.GRAPH_STYLING,
    )

    if is_scan:
        argument_nodes, argument_edges = graph_utils.get_scan_arguments(
            graph_id,
            parent_id,
            eqn.params["jaxpr"].jaxpr.invars,
            eqn.invars,
            eqn.params["num_consts"],
            eqn.params["num_carry"],
            show_avals,
        )
    else:
        argument_nodes, argument_edges = graph_utils.get_arguments(
            graph_id,
            parent_id,
            eqn.params["jaxpr"].jaxpr.constvars,
            eqn.params["jaxpr"].jaxpr.invars,
            eqn.invars,
            show_avals,
        )
    graph.add_subgraph(argument_nodes)

    for sub_eqn in eqn.params["jaxpr"].jaxpr.eqns:
        sub_graph, in_edges, out_nodes, out_edges, n = get_sub_graph(
            sub_eqn, graph_id, n, collapse_primitives, show_avals
        )
        if isinstance(sub_graph, pydot.Subgraph):
            graph.add_subgraph(sub_graph)
        else:
            graph.add_node(sub_graph)
        for edge in in_edges:
            graph.add_edge(edge)
        for node in out_nodes:
            graph.add_node(node)
        for edge in out_edges:
            graph.add_edge(edge)

    if is_scan:
        output_nodes, out_edges, out_nodes, id_edges = graph_utils.get_scan_outputs(
            graph_id,
            parent_id,
            eqn.params["jaxpr"].jaxpr.invars,
            eqn.params["jaxpr"].jaxpr.outvars,
            eqn.outvars,
            eqn.params["num_carry"],
            show_avals,
        )
    else:
        output_nodes, out_edges, out_nodes, id_edges = graph_utils.get_outputs(
            graph_id,
            parent_id,
            eqn.params["jaxpr"].jaxpr.invars,
            eqn.params["jaxpr"].jaxpr.outvars,
            eqn.outvars,
            show_avals,
        )

    graph.add_subgraph(output_nodes)
    for edge in id_edges:
        graph.add_edge(edge)

    return graph, argument_edges, out_nodes, out_edges, n


def get_scan(
    eqn: jax_core.JaxprEqn,
    parent_id: str,
    n: int,
    collapse_primitives: bool,
    show_avals: bool,
) -> sub_graph_return:
    graph, argument_edges, out_nodes, out_edges, n = expand_non_primitive(
        eqn,
        parent_id,
        n,
        collapse_primitives,
        show_avals,
        is_scan=True,
    )
    graph.set_label(f"scan ({eqn.params['length']})")
    return graph, argument_edges, out_nodes, out_edges, n


def get_while_branch(
    jaxpr: jax_core.Jaxpr,
    parent_id: str,
    parent_args: typing.List[jax_core.Var],
    parent_outvars: typing.List[jax_core.Var],
    label: str,
    n: int,
    show_avals: bool,
    collapse_primitives: bool,
) -> typing.Tuple[
    typing.Union[pydot.Subgraph, pydot.Node],
    typing.List[pydot.Edge],
    typing.List[pydot.Edge],
]:
    graph_id = f"cluster_{parent_id}_{label}"

    if collapse_primitives and not utils.contains_non_primitives(jaxpr.eqns):
        graph = pydot.Node(
            name=graph_id,
            label=label,
            **styling.FUNCTION_NODE_STYLING,
        )
        arg_edges = list()
        out_edges = list()

        for (var, p_var) in zip(jaxpr.invars, parent_args):
            # TODO: What does the underscore mean?
            if str(var)[-1] == "_":
                continue
            is_literal = isinstance(var, jax_core.Literal)
            if not is_literal:
                arg_edges.append(pydot.Edge(f"{parent_id}_{p_var}", graph_id))

        for (var, p_var) in zip(jaxpr.outvars, parent_outvars):
            if isinstance(var, jax_core.DropVar):
                continue
            out_edges.append(pydot.Edge(graph_id, f"{parent_id}_{p_var}"))

        return graph, arg_edges, out_edges, n
    else:
        graph = graph_utils.get_subgraph(graph_id, label)
        arg_nodes, outer_arg_edges = graph_utils.get_arguments(
            graph_id,
            parent_id,
            [],
            jaxpr.invars,
            parent_args,
            show_avals,
        )
        graph.add_subgraph(arg_nodes)

        for eqn in jaxpr.eqns:
            (
                sub_graph,
                arg_edges,
                out_nodes,
                out_edges,
                n,
            ) = get_sub_graph(eqn, graph_id, n, collapse_primitives, show_avals)
            if isinstance(sub_graph, pydot.Subgraph):
                graph.add_subgraph(sub_graph)
            else:
                graph.add_node(sub_graph)
            for edge in arg_edges:
                graph.add_edge(edge)
            for node in out_nodes:
                graph.add_node(node)
            for edge in out_edges:
                graph.add_edge(edge)

        out_nodes, outer_out_edges, _, id_edges = graph_utils.get_outputs(
            graph_id,
            parent_id,
            jaxpr.invars,
            jaxpr.outvars,
            parent_outvars,
            show_avals,
        )
        graph.add_subgraph(out_nodes)
        for e in id_edges:
            graph.add_edge(e)

        return graph, outer_arg_edges, outer_out_edges, n


def get_while(
    eqn: jax_core.JaxprEqn,
    parent_id: str,
    n: int,
    collapse_primitives: bool,
    show_avals: bool,
) -> sub_graph_return:

    while_graph_id = f"{parent_id}_while_{n}"
    while_graph = graph_utils.get_subgraph(f"cluster_{while_graph_id}", "while")
    n = n + 1

    n_cond_const = eqn.params["cond_nconsts"]
    n_body_const = eqn.params["body_nconsts"]
    cond_consts = eqn.invars[:n_cond_const]
    body_consts = eqn.invars[n_cond_const : n_cond_const + n_body_const]
    init_carry = eqn.invars[n_cond_const + n_body_const :]

    arg_edges = list()
    out_edges = list()

    for var in eqn.invars:
        arg_id = f"{while_graph_id}_{var}"
        is_literal = isinstance(var, jax_core.Literal)
        while_graph.add_node(
            graph_utils.get_arg_node(arg_id, var, show_avals, is_literal)
        )
        if not is_literal:
            arg_edges.append(pydot.Edge(f"{parent_id}_{var}", arg_id))

    cond_graph, cond_arg_edges, _, n = get_while_branch(
        eqn.params["cond_jaxpr"].jaxpr,
        while_graph_id,
        cond_consts + init_carry,
        eqn.outvars,
        "cond",
        n,
        show_avals,
        collapse_primitives,
    )
    for e in cond_arg_edges:
        while_graph.add_edge(e)

    body_graph, body_arg_edges, body_out_edges, n = get_while_branch(
        eqn.params["body_jaxpr"].jaxpr,
        while_graph_id,
        body_consts + init_carry,
        eqn.outvars,
        "body",
        n,
        show_avals,
        collapse_primitives,
    )
    for e in body_arg_edges:
        while_graph.add_edge(e)
    for e in body_out_edges:
        while_graph.add_edge(e)

    if isinstance(cond_graph, pydot.Subgraph):
        while_graph.add_subgraph(cond_graph)
    else:
        while_graph.add_node(cond_graph)
    if isinstance(body_graph, pydot.Subgraph):
        while_graph.add_subgraph(body_graph)
    else:
        while_graph.add_node(body_graph)

    for var in eqn.outvars:
        arg_id = f"{while_graph_id}_{var}"
        while_graph.add_node(graph_utils.get_out_node(arg_id, var, show_avals))
        if not isinstance(var, jax_core.DropVar):
            out_edges.append(pydot.Edge(arg_id, f"{parent_id}_{var}"))

    return while_graph, arg_edges, [], out_edges, n


def get_sub_graph(
    eqn: jax_core.JaxprEqn,
    parent_id: str,
    n: int,
    collapse_primitives: bool,
    show_avals: bool,
) -> sub_graph_return:
    """
    Generate a node/subgraph representing a function

    The returned node/subgraph is conditional on the function
    type. This function recursively walks nodes on the graph
    of this function to generate sub-graphs of sub-functions

    Parameters
    ----------
    eqn: jax._src.core.JaxprEqn
        JaxprEqn of the function
    parent_id: str
        ID of the
    n: int
        Integer used to generate unique ids for nodes, incremented
        as new nodes are added
    collapse_primitives: bool
        If `True` any subgraph only consisting of primitive
        functions is collapsed into a single node
    show_avals: bool
        If `True` the type of the data is shown on
        argument/variable nodes on the generated graph

    Returns
    -------
    (
        typing.Union[pydot.Node, pydot.Subgraph],
        typing.List[pydot.Edge],
        typing.List[pydot.Node],
        typing.List[pydot.Edge],
        int
    )
        Tuple containing:
            - Subgraph or node representing the function
            - List of edges that will connect a parent graph to the
              arguments of the function
            - List of nodes that should be added to a parent graph (i.e.
              outputs of this graph)
            - List of edges connecting the outputs of this node to
              parent graph
            - Updated incremented integer used to get unique node ids
    """

    if utils.is_not_primitive(eqn):
        if (
            utils.contains_non_primitives(eqn.params["jaxpr"].jaxpr.eqns)
            or not collapse_primitives
        ):
            return expand_non_primitive(
                eqn,
                parent_id,
                n,
                collapse_primitives,
                show_avals,
            )
        else:
            # Return a node representing a function
            return _get_node(
                eqn,
                parent_id,
                show_avals,
                n,
                False,
            )
    else:
        if eqn.primitive.name == "cond":
            # Return a conditional subgraph
            return get_conditional(eqn, parent_id, n, collapse_primitives, show_avals)
        elif eqn.primitive.name == "scan":
            return get_scan(
                eqn,
                parent_id,
                n,
                collapse_primitives,
                show_avals,
            )
        elif eqn.primitive.name == "while":
            return get_while(
                eqn,
                parent_id,
                n,
                collapse_primitives,
                show_avals,
            )
        else:
            # Return a primitive node
            return _get_node(
                eqn,
                parent_id,
                show_avals,
                n,
                True,
            )
