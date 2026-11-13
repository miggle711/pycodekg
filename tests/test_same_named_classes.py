"""Same-named classes in different files are told apart through imports (#169).

A base class, a typed local, a property read, a raised exception, an
instantiated class or a decorator is looked up through the using file's own
scopes and imports, so it links to the one class that file means rather than
to every class with that name.
"""

from kg_construction.kg.builder import RepoASTParser

FOO = "class Foo:\n    def m(self): pass\n    @property\n    def value(self): return 1\n"


def _build(tmp_path, files):
    for path, source in files.items():
        target = tmp_path / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(source)
    kg = RepoASTParser(max_workers=1).parse_repo("t/r", tmp_path)
    nodes = {n["id"]: n for n in kg["nodes"]}

    def where(node_id):
        meta = nodes[node_id]["metadata"]
        return f'{meta.get("filepath")}:{meta.get("qualname", nodes[node_id]["label"])}'

    def edges(relation, source_label):
        return sorted(
            (where(e["target"]), e["metadata"].get("confidence"))
            for e in kg["edges"]
            if e["relation"] == relation and nodes[e["source"]]["label"] == source_label
        )
    return edges


def test_base_class_is_the_imported_one(tmp_path):
    edges = _build(tmp_path, {
        "a.py": "class Base: pass\n", "b.py": "class Base: pass\n",
        "use.py": "from a import Base\nclass Child(Base): pass\n",
    })
    assert edges("inherits", "Child") == [("a.py:Base", "exact")]


def test_base_class_through_a_package_that_re_exports_it(tmp_path):
    # Django's models.Model: defined in models/base.py, re-exported by
    # models/__init__.py, and every app writes models.Model.
    edges = _build(tmp_path, {
        "pkg/__init__.py": "",
        "pkg/models/__init__.py": "from pkg.models.base import Model\n",
        "pkg/models/base.py": "class Model: pass\n",
        "other.py": "class Model: pass\n",
        "app.py": "from pkg import models\nclass Article(models.Model): pass\n",
    })
    assert edges("inherits", "Article") == [("pkg/models/base.py:Model", "exact")]


def test_base_class_through_a_relative_and_a_star_import(tmp_path):
    edges = _build(tmp_path, {
        "pkg/__init__.py": "",
        "pkg/shapes.py": "class Shape: pass\n",
        "pkg/everything.py": "from .shapes import *\n",
        "pkg/square.py": "from .everything import *\nclass Square(Shape): pass\n",
        "elsewhere.py": "class Shape: pass\n",
    })
    assert edges("inherits", "Square") == [("pkg/shapes.py:Shape", "exact")]


def test_base_class_defined_in_the_same_function(tmp_path):
    edges = _build(tmp_path, {
        "x.py": "class Person: pass\n",
        "y.py": "class Person: pass\n",
        "test_z.py": "def test_it():\n    class Person: pass\n    class Boss(Person): pass\n",
    })
    assert edges("inherits", "Boss") == [("test_z.py:test_it.Person", "exact")]


def test_dotted_base_that_can_not_be_followed_gets_no_edge(tmp_path):
    edges = _build(tmp_path, {
        "a.py": "class A:\n    class Meta: pass\n",
        "b.py": "class B:\n    class Meta: pass\n",
        "c.py": "class Form:\n    def f(self):\n        class Meta(self.form_class.Meta): pass\n",
    })
    assert edges("inherits", "Meta") == []


def test_typed_local_calls_the_imported_class_method(tmp_path):
    # Item 1: x = Foo(); x.m() goes to a.Foo.m only.
    edges = _build(tmp_path, {
        "a.py": FOO, "b.py": FOO,
        "use.py": "from a import Foo\ndef f():\n    x = Foo()\n    x.m()\n",
    })
    assert ("a.py:Foo.m", "qualified") in edges("calls", "f")
    assert not any(target.startswith("b.py") for target, _ in edges("calls", "f"))


def test_typed_local_of_an_unknown_same_named_class_gets_no_edge(tmp_path):
    # The class is chosen at runtime, and two files define one with that name.
    edges = _build(tmp_path, {
        "a.py": FOO, "b.py": FOO,
        "use.py": "class Mw:\n    def f(self):\n        x = self.Foo()\n        x.m()\n",
    })
    assert edges("calls", "f") == []


def test_self_property_and_operator_stay_on_the_own_class(tmp_path):
    # Item 2: each Foo.combine reads its own Foo.value and its own __or__.
    source = ("class Foo:\n    @property\n    def value(self): return 1\n"
              "    def __or__(self, other): return self\n"
              "    def combine(self, other):\n        self.value\n        return self | other\n")
    edges = _build(tmp_path, {"a.py": source, "b.py": source})
    accessed = edges("accesses", "combine")
    assert accessed == [("a.py:Foo.value", "qualified"), ("b.py:Foo.value", "qualified")]
    called = [t for t, _ in edges("calls", "combine")]
    assert called == ["a.py:Foo.__or__", "b.py:Foo.__or__"]


def test_imported_class_attribute_read(tmp_path):
    edges = _build(tmp_path, {
        "a.py": FOO, "b.py": FOO,
        "use.py": "from a import Foo\ndef f():\n    x = Foo()\n    return x.value\n",
    })
    assert edges("accesses", "f") == [("a.py:Foo.value", "qualified")]


def test_closure_self_call_resolves_on_the_method_class(tmp_path):
    # Item 5: cb's own self.m() goes to K.m, not Other.m.
    edges = _build(tmp_path, {
        "k.py": ("class K:\n    def m(self): pass\n"
                 "    def run(self):\n        def cb():\n            self.m()\n        return cb\n"),
        "o.py": "class Other:\n    def m(self): pass\n",
    })
    assert edges("calls", "cb") == [("k.py:K.m", "qualified")]


def test_closure_with_its_own_self_does_not_use_the_method_class(tmp_path):
    edges = _build(tmp_path, {
        "k.py": ("class K:\n    def m(self): pass\n"
                 "    def run(self):\n        def cb(self):\n            self.m()\n        return cb\n"),
        "o.py": "class Other:\n    def m(self): pass\n",
    })
    assert ("k.py:K.m", "qualified") not in edges("calls", "cb")


def test_raises_uses_and_decorated_by_follow_imports(tmp_path):
    edges = _build(tmp_path, {
        "a.py": "class Error(Exception): pass\nclass Widget: pass\ndef register(f): return f\n",
        "b.py": "class Error(Exception): pass\nclass Widget: pass\ndef register(f): return f\n",
        "use.py": ("import a\nfrom a import Error, register\n"
                   "class App:\n    def setup(self):\n        self.w = a.Widget()\n"
                   "@register\ndef f():\n    raise Error()\n"),
    })
    assert edges("raises", "f") == [("a.py:Error", "exact")]
    assert edges("decorated_by", "f") == [("a.py:register", "exact")]
    assert edges("uses", "App") == [("a.py:Widget", "exact")]


def test_uses_finds_a_class_defined_in_one_of_its_methods(tmp_path):
    edges = _build(tmp_path, {
        "forms.py": "class TestForm: pass\n",
        "test_views.py": ("class ViewTests:\n    def test_one(self):\n"
                          "        class TestForm: pass\n        TestForm()\n"),
    })
    assert edges("uses", "ViewTests") == [("test_views.py:ViewTests.test_one.TestForm", "exact")]


def test_runtime_assigned_name_is_not_a_class(tmp_path):
    edges = _build(tmp_path, {
        "a.py": "class Wrapper: pass\n", "b.py": "class Wrapper: pass\n",
        "c.py": ("class Creation:\n    def clone(self):\n"
                 "        Wrapper = type(self.conn)\n        return Wrapper()\n"),
    })
    assert edges("uses", "Creation") == []
