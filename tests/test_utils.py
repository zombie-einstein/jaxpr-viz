from jpviz.dot import utils


def test_id_map():

    m = utils.IdMap()
    assert m.get_next_label("a") == "a"
    assert m.get_next_label("b") == "b"
    assert m.get_next_label("a") == "a"

    m.i = 25
    assert m.get_next_label("z") == "z"
    assert m.get_next_label("foo") == "aa"
    assert m.get_next_label("bar") == "bb"
