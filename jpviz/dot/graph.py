# flake8: noqa: C901
import typing

import jax._src.core as jax_core
import pydot

from . import styling, utils

sub_graph_return = typing.Tuple[
    typing.Union[pydot.Node, pydot.Subgraph], typing.List, typing.List, typing.List, int
]


def _get_arg_node(
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


def _get_var_node(var_id: str, var: jax_core.Var, show_avals: bool) -> pydot.Node:
    return pydot.Node(
        name=var_id,
        label=utils.get_node_label(var, show_avals),
        **styling.VAR_STYLING,
    )


def _get_out_node(out_id: str, var: jax_core.Var, show_avals: bool) -> pydot.Node:
    return pydot.Node(
        name=out_id,
        label=utils.get_node_label(var, show_avals),
        **styling.OUT_ARG_STYLING,
    )


def _get_subgraph(graph_id: str, label: str) -> pydot.Subgraph:
    return pydot.Subgraph(
        graph_id,
        label=label,
        rank="same",
        **styling.GRAPH_STYLING,
    )


def _get_arguments(
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
        argument_nodes.add_node(_get_arg_node(arg_id, var, show_avals, is_literal))
        if not is_literal:
            argument_edges.append(pydot.Edge(f"{parent_id}_{p_var}", arg_id))

    return argument_nodes, argument_edges


def _get_outputs(
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
        out_graph.add_node(_get_out_node(arg_id, var, show_avals))
        out_edges.append(pydot.Edge(arg_id, f"{parent_id}_{p_var}"))
        out_nodes.append(_get_var_node(f"{parent_id}_{p_var}", p_var, show_avals))

    return out_graph, out_edges, out_nodes, id_edges


def get_conditional(
    conditional,
    parent_id: str,
    collapse_primitives: bool,
    show_avals: bool,
    n: int,
) -> sub_graph_return:

    cond_graph_id = f"{parent_id}_cond_{n}"
    cond_graph = _get_subgraph(f"cluster_{cond_graph_id}", "switch")
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
        new_nodes.append(_get_arg_node(cond_var_id, cond_var, show_avals, True))
    in_edges.append(pydot.Edge(cond_var_id, cond_node_id))

    for arg in conditional.invars[1:]:
        arg_id = f"{cond_graph_id}_{arg}"
        is_literal = isinstance(arg, jax_core.Literal)
        cond_arguments.add_node(_get_arg_node(arg_id, arg, show_avals, is_literal))
        in_edges.append(pydot.Edge(f"{parent_id}_{arg}", arg_id))

    cond_graph.add_subgraph(cond_arguments)

    for i, branch in enumerate(conditional.params["branches"]):
        if len(branch.eqns) == 0:
            branch_graph_id = f"{cond_node_id}_branch_{i}"
            if collapse_primitives:
                cond_graph.add_node(
                    pydot.Node(
                        name=branch_graph_id,
                        label=f"Branch {i}: Id",
                        **styling.FUNCTION_NODE_STYLING,
                    )
                )
                for var in branch.jaxpr.invars:
                    # TODO: What does the underscore mean?
                    if str(var)[-1] == "_":
                        continue
                    cond_graph.add_edge(
                        pydot.Edge(f"{cond_graph_id}_{var}", branch_graph_id)
                    )
                for var in conditional.outvars:
                    cond_graph.add_edge(
                        pydot.Edge(branch_graph_id, f"{cond_graph_id}_{var}")
                    )
            else:
                branch_graph = _get_subgraph(
                    f"cluster_{branch_graph_id}", f"Branch {i}"
                )
                for var, c_var in zip(branch.jaxpr.outvars, conditional.outvars):
                    arg_id = f"{branch_graph_id}_{var}"
                    branch_graph.add_node(_get_var_node(arg_id, var, show_avals))
                    cond_graph.add_edge(pydot.Edge(f"{cond_graph_id}_{var}", arg_id))
                    cond_graph.add_edge(pydot.Edge(arg_id, f"{cond_graph_id}_{c_var}"))
                cond_graph.add_subgraph(branch_graph)
        elif len(branch.eqns) == 1:
            (
                branch_graph,
                branch_in_edges,
                branch_out_nodes,
                branch_out_edges,
                n,
            ) = get_sub_graph(
                branch.eqns[0],
                cond_graph_id,
                n,
                collapse_primitives,
                show_avals,
            )
            branch_graph.set_label(f"Branch {i}: {branch_graph.get_label()}")
            if isinstance(branch_graph, pydot.Subgraph):
                cond_graph.add_subgraph(branch_graph)
            else:
                cond_graph.add_node(branch_graph)
            for edge in branch_in_edges:
                cond_graph.add_edge(edge)
            for edge, var in zip(branch_out_edges, conditional.outvars):
                cond_graph.add_edge(
                    pydot.Edge(edge.get_source(), f"{cond_graph_id}_{var}")
                )
        else:
            branch_graph_id = f"{cond_node_id}_branch_{i}"
            branch_label = f"Branch {i}: λ"
            if utils.contains_non_primitives(branch.eqns) or not collapse_primitives:
                branch_graph = _get_subgraph(f"cluster_{branch_graph_id}", branch_label)

                branch_args, arg_edges = _get_arguments(
                    branch_graph_id,
                    cond_graph_id,
                    branch.jaxpr.invars,
                    branch.jaxpr.invars,
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
                ) = _get_outputs(
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
                for var in branch.jaxpr.invars:
                    # TODO: What does the underscore mean?
                    if str(var)[-1] == "_":
                        continue
                    cond_graph.add_edge(
                        pydot.Edge(f"{cond_graph_id}_{var}", branch_graph_id)
                    )
                for var in conditional.outvars:
                    cond_graph.add_edge(
                        pydot.Edge(branch_graph_id, f"{cond_graph_id}_{var}")
                    )

    cond_out_graph, cond_out_edges, cond_out_nodes, _ = _get_outputs(
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
    x: jax_core.JaxprEqn,
    graph_id: str,
    show_avals: bool,
    n: int,
    is_primitive: bool,
) -> sub_graph_return:

    name = str(x.primitive) if is_primitive else x.params["name"]
    node_id = f"{name}_{n}"
    n = n + 1

    style = styling.PRIMITIVE_STYLING if is_primitive else styling.FUNCTION_NODE_STYLING
    node = pydot.Node(name=node_id, label=name, **style)

    new_nodes = list()
    in_edges = list()
    out_edges = list()

    for var in x.invars:
        if isinstance(var, jax_core.Literal):
            new_nodes.append(_get_arg_node(f"{graph_id}_{var}", var, show_avals, True))
        in_edges.append(pydot.Edge(f"{graph_id}_{var}", node_id))

    for var in x.outvars:
        new_nodes.append(
            pydot.Node(
                name=f"{graph_id}_{var}",
                label=utils.get_node_label(var, show_avals),
                **styling.VAR_STYLING,
            )
        )
        out_edges.append(pydot.Edge(node_id, f"{graph_id}_{var}"))

    return node, in_edges, new_nodes, out_edges, n


def get_sub_graph(
    eqn: jax_core.JaxprEqn,
    parent_id: str,
    n: int,
    collapse_primitives: bool,
    show_avals: bool,
) -> sub_graph_return:

    if utils.is_not_primitive(eqn):
        if (
            utils.contains_non_primitives(eqn.params["jaxpr"].jaxpr.eqns)
            or not collapse_primitives
        ):
            # Recursively return a subgraph
            graph_name = eqn.params["name"]
            graph_id = f"{graph_name}_{n}"
            n = n + 1

            graph = pydot.Subgraph(
                f"cluster_{graph_id}",
                rank="same",
                label=graph_name,
                **styling.GRAPH_STYLING,
            )

            argument_nodes, argument_edges = _get_arguments(
                graph_id,
                parent_id,
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

            output_nodes, out_edges, out_nodes, id_edges = _get_outputs(
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
            return get_conditional(eqn, parent_id, collapse_primitives, show_avals, n)
        else:
            # Return a primitive node
            return _get_node(
                eqn,
                parent_id,
                show_avals,
                n,
                True,
            )
