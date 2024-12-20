import typing

from jax._src import core as jax_core


class IdMap:
    def __init__(self):
        self.i = 0
        self.map = dict()
        self.chars = [chr(i) for i in range(97, 123)]
        self.n_chars = len(self.chars)

    def get_next_label(self, node_id: str) -> str:
        if node_id in self.map:
            return self.map[node_id]
        else:
            i = self.i % self.n_chars
            j = (self.i - i) // self.n_chars
            label = (j + 1) * self.chars[i]
            self.map[node_id] = label
            self.i += 1
            return label


def get_node_label(
    v: typing.Union[jax_core.Var, jax_core.Literal], show_avals: bool, id_map: IdMap
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
    id_map: IdMap
        Node id to label mapping

    Returns
    -------
    str
    """
    if isinstance(v, jax_core.Literal):
        if show_avals:
            return f"{v}: {v.aval.str_short()}"
        else:
            return str(v)
    else:
        label = id_map.get_next_label(str(id(v)))
        if show_avals:
            return f"{label}: {v.aval.str_short()}"
        else:
            return label


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
    return x.primitive.name == "pjit"


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
    return any(
        [
            ("jaxpr" in e.params or e.primitive.name in {"cond", "scan", "while"})
            for e in eqns
        ]
    )
