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


def contains_non_primitives(eqns: typing.List[jax_core.JaxprEqn]) -> bool:
    """
    Check it the sub-functions of a JaxPR contains only JAX primitives

    Parameters
    ----------
    eqns: List[jax._src.core.JaxprEqn]
        List of JaxprEqns

    Returns
    -------
    bool:
        `True` if any of the sub-eqns are non-primitive
    """
    return any([("jaxpr" in e.params or e.primitive.name == "cond") for e in eqns])
