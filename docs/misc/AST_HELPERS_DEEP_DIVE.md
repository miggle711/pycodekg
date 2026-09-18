# `ast/helpers.py` deep dive

A complete walkthrough of the pure AST-manipulation toolkit that `builder.py`
and `context.py` both call into. No I/O, no multiprocessing, no KG dataclass
dependencies -- every function here is a straightforward AST-in, data-out
transformation, safe to import from any layer.

The file's own docstring lists six sections in file order; this doc follows
the same organization.

## Identity & path helpers

**`_make_id(text)`** -- MD5 hash of a qualified-name string, truncated to 8
hex chars. This is what makes node IDs deterministic: the same entity
(same repo + path + name) always produces the same ID, across different
commits or different `build()` calls, without needing any persistent
counter or database.

**`_is_test_file(filepath)`** -- convention-based: filename starts with
`test_`, ends with `_test.py`, or `tests`/`test` appears anywhere in the
path parts. Used both to decide a file's own node `type`
(`file`/`test_file`) and to gate `assert_patterns` extraction (see
`_build_func_metadata` below) to test files only.

## AST unparsing

**`_safe_unparse(node)`** -- wraps `ast.unparse` in a try/except, returning
`None` on failure instead of raising. `ast.unparse` can fail on malformed
or unusual AST nodes (e.g. from macro-generated code); this is what keeps
the whole parse pipeline running even when one decorator, signature, or
base-class expression can't be rendered back to text. Used everywhere
source-text-like output is needed: decorator strings, signature
annotations/defaults, raised-exception expressions, base class names.

## Call-site extraction

**`_extract_callee_name(call)`** -- from an `ast.Call` node, returns the
bare callee name: `foo()` -> `'foo'`, `obj.method()` -> `'method'`.
Explicitly documented as NOT handling subscript calls (`obj['key']()`) or
chained calls (`foo()()`) -- a deliberate scope limit (avoiding false
positives from unreliable inference), not an oversight.

**`_extract_call_receiver(call)`** -- for an attribute call, returns the
receiver expression as a string: `obj.method()` -> `'obj'`,
`json.loads()` -> `'json'`, `self.foo.bar()` -> `'self.foo'`. This is what
lets `builder.py`'s pass-2 resolution tell `json.loads` apart from
`pickle.loads`, and `self.method()` apart from any other class's
same-named method.

**`_annotated_param_types(func_node)`** -- maps parameter names to their
bare annotated type, but ONLY for direct `Name`/`Attribute` annotations
(`x: Request` or `x: pkg.Request` -> `'Request'`). Subscripted annotations
like `Optional[Request]` or `List[Request]` are deliberately skipped: the
parameter could be `None` or a container, so the "this receiver is an
instance of this exact type" assumption that `_collect_local_types` relies
on wouldn't hold.

**`_extract_property_accesses(func_node)`** -- finds genuine attribute
READS (`obj.attr`, no parentheses), which is the ONLY signal that exists
for a `@property`-decorated method, since a property read never appears
as an `ast.Call` node at all. Three things are deliberately excluded: an
`Attribute` node that's actually the `.func` of an enclosing `Call` (that's
a real call, already covered by call-edge extraction, not a property
read); anything in `Store`/`Del` context (an assignment target, a write,
not a read); and anything inside a nested function's own scope (its
accesses belong there, not to the enclosing function). Returns
deduplicated `(attr_name, receiver)` pairs.

**`_collect_local_types(func_node, before_line=None)`** -- infers a local
variable's class from either a parameter annotation or a visible
constructor call (`x = SomeClass(...)`). The constructor-call heuristic is
narrow and deliberate: only a capitalized bare name (`SomeClass(...)`) or
a capitalized final attribute component (`pkg.SomeClass(...)`) counts,
matching the PEP 8 class-naming convention. A LATER reassignment of the
same name overwrites an earlier one -- which is exactly why
`before_line` exists: a caller resolving "what type did `x` hold AT this
specific call site" (not "what's the last type `x` is ever assigned across
the whole function") must pass that call site's own line number, since
otherwise a later reassignment elsewhere in the function would be
indistinguishable from the type in effect earlier. Nested function bodies
are skipped for assignments (their locals belong to that scope).

## Function/method metadata

**`_get_docstring(node)`** -- the first statement's string value, if it's
a string constant expression; works for function, class, OR module nodes
(all share the same `.body[0]` shape when a docstring is present).

**`_get_decorators(node)`** -- every decorator, unparsed to a string:
plain names (`@staticmethod`), attribute access (`@pytest.mark.skip`), and
decorator calls (`@pytest.mark.parametrize("x", [1,2,3])`) all handled
uniformly via `_safe_unparse`.

**`_get_signature(node)`** -- the full parameter list in declaration order:
positional-only + regular args first, then keyword-only, then `*args`/
`**kwargs` last (prefixed accordingly). The one genuinely tricky part:
`defaults` in Python's AST is RIGHT-aligned against the arg list (i.e.
`def f(a, b=1)` has `defaults=[1]` aligned to `b`, not `a`) -- handled by
left-padding with `None` so each arg's index lines up correctly with its
own default (or lack of one).

**`_get_signature_and_source_text(func_node, source_lines)`** -- slices
the function's own `def` line(s) and full body text directly out of the
enclosing file's source, via the AST node's real line range
(`func_node.lineno`/`end_lineno`). Two LLM-facing convenience fields
(`signature`, `source_code`) -- distinct from `params`, which is
structured data. The signature slice specifically stops before the first
body statement's own line, so a multi-line wrapped signature is captured
in full without pulling in any of the function's actual body. Both fields
are `""` if `source_lines` isn't given at all (a caller with no file text
handy) -- documented as never having been populated before this was added,
which meant the KG-augmented arm's prompt showed a docstring and a
param-name list but never the actual code being tested (several real
test-quality bugs traced back to exactly this gap).

**`_get_exceptions(node)`** -- walks the ENTIRE function body (`ast.walk`,
not just direct children) collecting `raise` expressions and `except`
handler types, so re-raises and chained handlers nested arbitrarily deep
are all captured, deduplicated.

**`_extract_conditions(node)`** -- boundary-condition extraction from
`if`/`while`/`assert` statements, explicitly stopping at nested function
boundaries (a helper function defined inside the target function shouldn't
have its own conditions attributed to the outer function). Deduplicated
by `(type, lineno)`.

**`_extract_data_flows(node)`** -- three things in one pass: `returns`
(unparsed return-value expressions), `mutates_attributes` (`self.x = ...`
or `self.x[key] = ...`, mapped to the values assigned), and
`parameter_usage` (line numbers where each parameter name appears as a
`Load`). A real subtlety: nested function bodies ARE walked for parameter
usage (since a closure's parameter references are still meaningful "this
parameter is used" signal) but NOT for mutations/returns (those belong to
the nested function's own semantics, not the outer one's).

**`_count_branches(node)`** -- counts `if`/`for`/`while` statements,
scoped to this function only (a nested function's own branches don't
count toward the outer function's branch count).

**`_get_assert_patterns(node)`** -- captures both plain `assert`
statements and unittest-style `self.assertEqual(...)`-shaped calls (any
call whose attribute name starts with `assert`). Only ever populated for
test files -- gated at the `_build_func_metadata` call site, not here --
to avoid noise from invariant asserts in production code.

## Class metadata

**`_get_base_names(class_node)`** -- unparsed base-class expressions:
`class Foo(Bar, Mixin)` -> `['Bar', 'Mixin']`; `class Foo(pkg.Base)` ->
`['pkg.Base']`.

**`_get_class_attributes(class_node)`** -- instance attributes assigned in
`__init__` ONLY (both plain `self.x = ...` and annotated `self.x: int =
...`) -- other methods are deliberately not scanned, since an attribute
set only conditionally in some other method may not reliably exist on
every instance.

**`_get_instantiated_classes_in_class(class_node)`** -- aggregates
`_get_instantiated_classes` (below) across every method of a class, for
the class-level `uses` edge (`builder.py` emits one `uses` edge per
distinct class instantiated anywhere in any of this class's methods).

## Function-body analysis

**`_get_attribute_accesses(func_node, class_name=None)`** -- `self.x`
reads vs. writes, in one pass. A read is any `self.attr` access not
already established as a write target on the same visit; a write is any
`self.attr = ...`/`self.attr: T = ...` assignment (plain or subscripted:
`self.x[key] = ...` still counts as a write to `x`). Nested function
scopes are excluded (a closure's own `self` accesses, if it captures
`self` at all, belong to its own analysis, not the enclosing method's).

**`_get_used_imports(func_node, import_map)`** -- which imported names are
actually referenced inside this specific function body, mapped back to
their fully-qualified path via the file-level `import_map`. This is what
becomes `depends_on` edges in `builder.py`, and what `_build_func_metadata`
uses (via `_get_annotation_type_names` + this) to populate `external_deps`
-- telling a test generator exactly what to mock.

**`_get_instantiated_classes(func_node)`** -- the uppercase-heuristic:
finds `Call` nodes whose callee is a capitalized bare name or has a
capitalized final attribute (`SomeClass(...)` or `pkg.SomeClass(...)`),
per PEP 8's class-naming convention. This heuristic's known blind spot
(a LOWERCASE factory function that itself returns a class instance, e.g.
`session = requests.session()`) is exactly what
`_get_factory_call_sites` exists to cover, via the optional pyright pass.

**`_get_factory_call_sites(func_node)`** -- the deliberate complement of
the function above: finds assignments from LOWERCASE-named calls
(`x = some_call(...)`, `self.x = some_call(...)`, `other.x =
some_call(...)`), recording each site's exact `(line, col)` position so an
external type checker (`kg/type_inference.py`) can be asked, LSP-style,
"what type does this expression actually evaluate to." Only simple `Name`
targets and single-level attribute targets are recorded -- tuple targets
or deeper attribute chains don't reduce to a single position an
editor-style hover query resolves cleanly, so they're skipped rather than
guessed at. The returned `receiver` is `None` for a bare `Name` target or
a `self.x = ...` target (the enclosing class is already known without
needing pyright at all); for any other `receiver.x = ...` target,
`receiver` is that name, letting the caller resolve ITS type too (via
`_collect_local_types`) to know which class the resulting `uses` edge
should actually originate from.

## Aggregators

**`_get_annotation_type_names(func_node)`** -- walks every parameter
annotation AND the return annotation, extracting the bare final name
component from any `Name`/`Attribute` node found anywhere inside (so
`Optional[MyClass]` yields both `'Optional'` and `'MyClass'`, not just the
outer subscript). Feeds `external_deps` in `_build_func_metadata`.

**`_build_func_metadata(func_node, rel_path, repo, parent_class=None,
parent_function=None, import_map=None, source_lines=None)`** -- the
central aggregator that assembles a function/method node's FULL metadata
dict, centralizing what would otherwise be duplicated across
`builder.py`'s top-level-function, class-method, and nested-function
branches (and reused directly by `context.py`'s synthetic new-file seed
builder, so a node built either way has identical shape). `parent_class`
and `parent_function` are mutually exclusive in practice -- a nested
function's own enclosing scope is either a class method or a plain
function, never both at once. Composes essentially every function above:
signature, exceptions, source slicing, external deps (annotation types +
instantiated classes that resolve through `import_map`), side effects
(attribute writes), data flows, conditions, and -- gated to test files
only -- assert patterns.

**`_collect_file_level_info(tree)`** -- a single combined pass (avoiding
three separate tree walks) collecting: `import_map` (`{local_name:
fully_qualified_name}`, handling both `import os.path` -> local name
`os` and `from x import y as z` -> local name `z`), `__all__` exports (if
the module defines one, as a list/tuple of string constants), and
module-level UPPER_CASE constants (both plain and annotated assignment
forms). `import_map` in particular is threaded through nearly every other
function in this file that needs to resolve a bare name back to its real
module origin.
