"""`X.as_view()(request)` reaches the view's HTTP handlers (#119).

Django's View.as_view() returns a function that builds an X and calls its
get, post, ... method. A test that only writes `X.as_view()(request)` has no
literal call of X.get, so the call graph used to stop at View.as_view.
"""

import textwrap
from pathlib import Path

import pytest

from kg_construction.kg.builder import RepoASTParser

BASE = """
    class View:
        @classmethod
        def as_view(cls, **kwargs): ...
"""


def _handlers(tmp_path: Path, files: dict):
    """(handler qualname, confidence) for the calls edges made through as_view."""
    for name, source in files.items():
        (tmp_path / name).write_text(textwrap.dedent(source))
    kg = RepoASTParser(max_workers=1).parse_repo("test/repo", tmp_path)
    nodes = {n["id"]: n for n in kg["nodes"]}
    return sorted(
        (nodes[e["target"]]["metadata"]["qualname"], e["metadata"]["confidence"])
        for e in kg["edges"]
        if e["relation"] == "calls" and e["metadata"].get("via") == "as_view"
    )


CALLER = "from views import {name}\ndef test_it(request):\n    {name}.as_view()(request)\n"


@pytest.mark.parametrize("views, name, expected", [
    # One handler is certain enough to be qualified.
    ("class A(View):\n    def get(self, request): ...\n", "A", [("A.get", "qualified")]),
    # Several handlers: which one runs depends on the request.
    ("class A(View):\n    def get(self, request): ...\n    def post(self, request): ...\n", "A",
     [("A.get", "name_only"), ("A.post", "name_only")]),
    # A handler the view inherits from a parent in the repo.
    ("class Base(View):\n    def get(self, request): ...\nclass A(Base): ...\n", "A", [("Base.get", "qualified")]),
    # Not a view with handlers, or not a class in the repo: nothing to add.
    ("class A(View):\n    def helper(self): ...\n", "A", []),
    ("class A(View):\n    def get(self, request): ...\n", "Missing", []),
], ids=["one_handler", "several_handlers", "inherited_handler", "no_handlers", "unknown_class"])
def test_as_view_reaches_the_views_handlers(tmp_path, views, name, expected):
    files = {"base.py": BASE, "views.py": "from base import View\n" + views, "test_views.py": CALLER.format(name=name)}
    if name == "Missing":
        files["test_views.py"] = "def test_it(request):\n    Missing.as_view()(request)\n"
    assert _handlers(tmp_path, files) == expected


def test_the_request_method_the_caller_builds_picks_the_handler(tmp_path):
    files = {
        "base.py": BASE,
        "views.py": "from base import View\nclass A(View):\n    def get(self, request): ...\n    def post(self, request): ...\n",
        "test_views.py": "from views import A\ndef test_it(rf):\n    request = rf.post('/x/')\n    A.as_view()(request)\n",
    }
    assert _handlers(tmp_path, files) == [("A.post", "qualified")]


def test_same_named_views_are_told_apart_by_the_import(tmp_path):
    files = {
        "base.py": BASE,
        "admin_views.py": "from base import View\nclass Index(View):\n    def get(self, request): ...\n",
        "site_views.py": "from base import View\nclass Index(View):\n    def post(self, request): ...\n",
        "test_views.py": "from site_views import Index\ndef test_it(request):\n    Index.as_view()(request)\n",
    }
    assert _handlers(tmp_path, files) == [("Index.post", "qualified")]


ITEM_VIEW = """
    from base import View
    class ItemView(View):
        def get(self, request): ...
        def post(self, request): ...
"""


@pytest.mark.parametrize("body, expected", [
    # rf builds the request, so only post is meant. The dict's .get(...) does not count.
    ('def test_it(rf):\n    options = {}\n    options.get("timeout")\n'
     '    ItemView.as_view()(rf.post("/item/"))\n', [("ItemView.post", "qualified")]),
    # Only a dict lookup: no request is built, so the handler stays unknown (both).
    ('def test_it():\n    cfg = {}\n    cfg.get("x")\n    ItemView.as_view()(object())\n',
     [("ItemView.get", "name_only"), ("ItemView.post", "name_only")]),
    # The test client, with a reverse() url.
    ('def test_it(client):\n    client.post(reverse("item"))\n    ItemView.as_view()\n',
     [("ItemView.post", "qualified")]),
    # A url path as the first argument, whatever the receiver is called.
    ('def test_it(c):\n    c.get("/item/")\n    ItemView.as_view()\n', [("ItemView.get", "qualified")]),
    # An HTTP library or a DataFrame is not a request builder.
    ('import requests\ndef test_it(df):\n    requests.get("https://example.com")\n    df.head()\n'
     '    ItemView.as_view()\n', [("ItemView.get", "name_only"), ("ItemView.post", "name_only")]),
])
def test_only_a_call_that_builds_a_request_narrows_the_handlers(tmp_path, body, expected):
    files = {"base.py": BASE, "views.py": ITEM_VIEW,
             "test_views.py": "from views import ItemView\n" + body}
    assert _handlers(tmp_path, files) == expected
