import typing

from jax._src import core as jax_core


def get_node_label(
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


def is_not_primitive(x: jax_core.JaxprEqn) -> bool:
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


def contains_non_primitives(x: jax_core.JaxprEqn) -> bool:
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
