"""The #118 census must not confuse folded edges with individual call sites."""

import importlib.util
from pathlib import Path
from textwrap import dedent

SPEC = importlib.util.spec_from_file_location(
    "measure_chained_receivers", Path(__file__).resolve().parents[1] / "scripts/measure_chained_receivers.py")
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


def sites(tmp_path, files, filepath="mod.py", qualname="caller"):
    for name, text in files.items():
        (tmp_path / name).write_text(dedent(text))
    parser = audit.ObservingParser({(filepath, qualname)}, max_workers=1)
    kg = parser.parse_repo("audit/repo", tmp_path)
    specs = [{"status": "matched", "instance": "instance-1", "filepath": filepath, "head_qualname": qualname}]
    return audit.collect_sites(parser, kg, specs, audit.SourceIndex(tmp_path))


CLASS_SOURCE = '''\
class QuerySet:
    def filter(self): return self
    def delete(self): pass
class Other:
    def delete(self): pass
'''


def test_missing_chain_is_counted_even_when_a_direct_call_has_a_same_named_edge(tmp_path):
    result = sites(tmp_path, {"mod.py": CLASS_SOURCE + '''
def caller():
    q = QuerySet()
    q.delete()
    q.filter().delete()
'''})
    assert len(result) == 1
    assert result[0]["status"] == "unresolved"
    assert result[0]["same_name_edge_elsewhere_in_caller"]
    assert [(c["evidence"], c["qualname"]) for c in result[0]["candidates"]] == [("return_self", "QuerySet.delete")]


def test_annotation_alias_is_resolved_in_the_factory_definition_file(tmp_path):
    result = sites(tmp_path, {
        "result.py": '''class Result:
    def save(self): pass
class Other:
    def save(self): pass
''',
        "factory.py": '''from result import Result as Alias
def make() -> "Alias": return Alias()
''',
        "mod.py": '''import factory
def caller(): factory.make().save()
''',
    })
    assert len(result) == 1 and result[0]["status"] == "unresolved"
    assert [(c["evidence"], c["qualname"]) for c in result[0]["candidates"]] == [("return_annotation", "Result.save")]


def test_unknown_return_and_async_result_are_not_claimed_resolvable(tmp_path):
    result = sites(tmp_path, {"mod.py": '''class Result:
    def save(self): pass
class Other:
    def save(self): pass
def choose(flag): return Result() if flag else Other()
async def make() -> Result: return Result()
def caller(flag):
    choose(flag).save()
    make().save()
'''})
    assert len(result) == 2
    assert all(r["status"] == "unresolved" and r["candidates"] == [] for r in result)
    assert any("coroutine" in note for note in result[1]["notes"])


def test_nested_function_and_lambda_chains_are_outside_target_own_body(tmp_path):
    result = sites(tmp_path, {"mod.py": CLASS_SOURCE + '''
def caller():
    q = QuerySet()
    def nested(): q.filter().delete()
    callback = lambda: q.filter().delete()
    q.filter().delete()
'''})
    assert len(result) == 1
    assert result[0]["call"] == "q.filter().delete()"


def test_rebinding_self_does_not_count_as_identity_preserving_return(tmp_path):
    result = sites(tmp_path, {"mod.py": '''class QuerySet:
    def filter(self):
        self = Other()
        return self
    def delete(self): pass
class Other:
    def delete(self): pass
def caller():
    q = QuerySet()
    q.filter().delete()
'''})
    assert len(result) == 1 and result[0]["status"] == "unresolved"
    assert result[0]["candidates"] == []


def test_missing_head_target_is_reported_instead_of_counted_as_zero_sites(tmp_path):
    (tmp_path / "mod.py").write_text("def replacement(): pass\n")
    row = {"id": "instance-1", "code_file": "mod.py", "patch": "unused fixture",
           "code_src": "def removed():\n    return 1\n"}
    specs, totals = audit.target_specs([row], lambda *args: [(1, 2)], audit.SourceIndex(tmp_path))
    assert totals["target_function_occurrences"] == 1
    assert specs[0]["status"] == "head_target_missing"


def test_an_ambiguous_head_redefinition_is_not_assigned_by_ordinal(tmp_path):
    (tmp_path / "mod.py").write_text("if flag:\n    def f(): pass\nelse:\n    def f(): pass\n")
    row = {"id": "instance-1", "code_file": "mod.py", "patch": "unused fixture", "code_src": "def f(): pass\n"}
    specs, _ = audit.target_specs([row], lambda *args: [(1, 1)], audit.SourceIndex(tmp_path))
    assert specs[0]["status"] == "head_target_ambiguous"


def test_nested_chain_calls_with_the_same_start_position_remain_distinct(tmp_path):
    result = sites(tmp_path, {"mod.py": """class Result:
    def step(self): return self
    def save(self): pass
class Other:
    def step(self): return self
    def save(self): pass
def make() -> Result: return Result()
def caller(): make().step().save()
"""})
    assert len(result) == 2
    assert {r["outer"] for r in result} == {"step", "save"}
    assert len({(r["line"], r["column"]) for r in result}) == 1
    assert len({r["end_column"] for r in result}) == 2
