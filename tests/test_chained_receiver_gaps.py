"""#118 investigations: record gaps without changing resolution or guessing types."""

import pytest

from kg_construction.kg.builder import RepoASTParser

SOURCE = '''class Result:
    def save(self): pass

class OtherResult:
    def save(self): pass

def make_result() -> Result:
    return Result()

def unknown_result(flag):
    return Result() if flag else OtherResult()

class QuerySet:
    def filter(self): return self
    def delete(self): pass

class OtherQuery:
    def delete(self): pass

class Form:
    @property
    def media(self): return "form"

class OtherForm:
    @property
    def media(self): return "other"

class FormSet:
    @property
    def empty_form(self) -> Form: return Form()

def direct():
    result = Result()
    result.save()

def annotated_chain():
    make_result().save()

def unknown_chain(flag):
    unknown_result(flag).save()

def self_chain():
    query = QuerySet()
    query.filter().delete()

def property_chain():
    forms = FormSet()
    return forms.empty_form.media
'''


@pytest.fixture
def graph(tmp_path):
    (tmp_path / "mod.py").write_text(SOURCE)
    return RepoASTParser(max_workers=1).parse_repo("test/repo", tmp_path)


def targets(graph, caller, relation):
    nodes = {n["id"]: n for n in graph["nodes"]}
    source = next(n["id"] for n in graph["nodes"]
                  if n["metadata"].get("qualname") == caller)
    return {nodes[e["target"]]["metadata"].get("qualname") for e in graph["edges"]
            if e["source"] == source and e["relation"] == relation}


@pytest.mark.xfail(strict=True, raises=AssertionError,
                   reason="#118: the return annotation is not propagated to a call-result receiver")
def test_call_on_annotated_factory_result_reaches_the_declared_method(graph):
    assert "Result.save" in targets(graph, "annotated_chain", "calls")


@pytest.mark.xfail(strict=True, raises=AssertionError,
                   reason="#118: a method returning self does not type the next receiver")
def test_call_on_method_returning_self_reaches_the_same_class(graph):
    assert "QuerySet.delete" in targets(graph, "self_chain", "calls")


def test_call_on_unknown_return_type_is_not_guessed(graph):
    # The factory can return either class. No principled unique save target exists.
    assert not {"Result.save", "OtherResult.save"} & targets(graph, "unknown_chain", "calls")
    assert "unknown_result" in targets(graph, "unknown_chain", "calls")


@pytest.mark.xfail(strict=True, raises=AssertionError,
                   reason="#118: a property's return annotation does not type the next property receiver")
def test_property_chain_reaches_the_outer_property(graph):
    assert "Form.media" in targets(graph, "property_chain", "accesses")


def test_property_chain_already_reaches_its_inner_property(graph):
    assert "FormSet.empty_form" in targets(graph, "property_chain", "accesses")


def test_direct_typed_receiver_already_resolves(graph):
    assert "Result.save" in targets(graph, "direct", "calls")
