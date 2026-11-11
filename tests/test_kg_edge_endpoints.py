"""Every edge of a built KG connects two real nodes (#225).

An import node and the depends_on edge to it are built in different places,
so a change to how one is named can leave the other pointing at a node that
does not exist. KGValidator now reports such edges; this builds a repo with
the import shapes that exist and checks the validator is satisfied.
"""

import textwrap
from pathlib import Path

from kg_construction.kg.builder import RepoASTParser
from kg_construction.kg.validator import KGValidator

FILES = {
    "pkg/__init__.py": "",
    "pkg/utils.py": "def helper(): ...\nclass Base: ...\n",
    "pkg/sub/__init__.py": "",
    "pkg/sub/leaf.py": "from ..utils import helper\nfrom .. import utils\ndef g():\n    helper()\n    return utils\n",
    "pkg/a.py": """
        from __future__ import annotations
        import os
        import os.path
        from typing import TYPE_CHECKING
        from . import utils
        from .utils import helper, Base
        from .sub.leaf import g
        if TYPE_CHECKING:
            from .utils import Base as TypedBase
        try:
            import json as codec
        except ImportError:
            codec = None

        class K(Base):
            def run(self) -> TypedBase:
                import re
                helper()
                g()
                utils.helper()
                return os.path.join(re.escape("a"), codec.dumps({}))
    """,
}


def test_every_edge_connects_two_nodes(tmp_path: Path):
    for name, text in FILES.items():
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(textwrap.dedent(text))
    kg = RepoASTParser(max_workers=1).parse_repo("test/repo", tmp_path)

    node_ids = {n["id"] for n in kg["nodes"]}
    dangling = [(e["relation"], e["source"], e["target"]) for e in kg["edges"]
                if e["source"] not in node_ids or e["target"] not in node_ids]
    assert dangling == []
    assert KGValidator(kg).validate()[0] is True
