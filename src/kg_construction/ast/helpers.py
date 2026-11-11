"""
ast_helpers.py

Pure AST-in, data-out helpers used by repo_kg_builder to extract node
metadata and edge targets from Python source trees. No external I/O,
no multiprocessing, no KG dataclass dependencies — safe to import from
any layer of the builder.

Organised into sections (in file order):
    - Identity & path helpers           (_make_id, _is_test_file)
    - AST unparsing                     (_safe_unparse)
    - Call-site extraction              (_extract_callee_name, _extract_call_receiver,
                                         _extract_property_accesses, _collect_local_types)
    - Function/method metadata          (_get_docstring, _get_decorators, _get_signature,
                                         _get_exceptions, _extract_conditions, _extract_data_flows,
                                         _count_branches, _get_assert_patterns)
    - Class metadata                    (_get_base_names, _get_class_attributes,
                                         _get_instantiated_classes_in_class)
    - Function-body analysis            (_get_attribute_accesses, _get_used_imports,
                                         _get_instantiated_classes)
    - Aggregators                       (_build_func_metadata, _collect_file_level_info)
"""

import ast
import hashlib
from pathlib import Path
from typing import List, Dict, FrozenSet, NamedTuple, Set, Optional, Tuple, Union


# ---------------------------------------------------------------------------
# Identity & path helpers
# ---------------------------------------------------------------------------

def split_source_lines(source: str) -> List[str]:
    """Split source into the lines `ast` numbers, keeping line endings.

    Splits after \\n only. str.splitlines also splits on form feeds and other
    separators that `ast` does not count, which would shift every later line
    number (#157). Source has already been through universal newline
    translation, so \\n is the only line ending left.
    """
    lines = [line + '\n' for line in source.split('\n')]
    lines[-1] = lines[-1][:-1]
    return lines if lines[-1] else lines[:-1]


def _make_id(text: str) -> str:
    """Generate a deterministic 16-character ID from a qualified name string.

    Uses MD5 truncated to 16 hex chars (64 bits). Raised from 8 (32 bits):
    at django's real scale (46,520 nodes) collision probability was ~25%,
    not negligible. A collision silently drops a node (see
    _aggregate_and_index's dedup). At 16 chars it's ~6e-11.
    """
    return hashlib.md5(text.encode()).hexdigest()[:16]


def _is_test_file(filepath: str) -> bool:
    """Return True if the file path matches common Python test file conventions.

    A `tests` directory anywhere in the path counts. A `test` or `testing`
    directory counts only at the repository root: a deeper one is usually a
    framework package (django/test/, sympy/testing/) and not tests. Nested
    test directories such as Lib/test/ are therefore typed as normal files.
    conftest.py is pytest configuration for tests, wherever it is.
    """
    parts = Path(filepath).parts
    name = Path(filepath).name
    return (
        name.startswith('test_') or
        name.endswith('_test.py') or
        name == 'conftest.py' or
        'tests' in parts or
        parts[:1] in (('test',), ('testing',))
    )


_DEF_TYPES = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)

# Nodes a def can be nested under without changing scope: statements
# (if/try/with/for/while/match...) plus the two non-stmt containers that
# hold statement bodies. Expressions are skipped on purpose, they can't
# contain a def, and descending into them would risk RecursionError on
# very deep expressions (e.g. a long chain of `a + b + c + ...`).
_TRY_TYPES = (ast.Try,) + ((ast.TryStar,) if hasattr(ast, 'TryStar') else ())

_SCOPE_TRANSPARENT = (ast.stmt, ast.ExceptHandler) + (
    (ast.match_case,) if hasattr(ast, 'match_case') else ()
)


def iter_scope_defs(node: ast.AST):
    """Yield the function/class defs that belong directly to `node`'s scope,
    in source order.

    Looks through compound statements (`if`/`try`/`with`/`for`/`while`/
    `match`), which don't introduce a scope, so a def under
    `if sys.version_info[0] == 2:` still belongs to the enclosing module/
    class/function. Stops at each def it finds and does not look inside it:
    a def's own body is a new scope, handled by calling this again on that
    def (see assign_qualnames).
    """
    for child in ast.iter_child_nodes(node):
        if isinstance(child, _DEF_TYPES):
            yield child
        elif isinstance(child, _SCOPE_TRANSPARENT):
            yield from iter_scope_defs(child)


def assign_qualnames(tree: ast.AST) -> List[Tuple[ast.AST, str]]:
    """Give every function/class def in `tree` a qualified name that is
    unique within the file, as (def_node, qualname) pairs.

    A qualname is the dotted path to the def (`Klass.method`,
    `outer.inner`). The first def with a given path keeps it as is; each
    later one with the same path (a property getter and its `@x.setter`, the
    two branches of an `if/else`, `@overload` stubs) gets an ordinal suffix
    in source order: `Klass.x`, `Klass.x#2`, `Klass.x#3`. `#` can't occur in
    a Python identifier, so a suffix never collides with a real name.
    Children are prefixed with their parent's *unique* qualname
    (`Klass#2.method`), so the members of a redefined class can't collide
    with the original's.

    The result is deterministic (source order) and is shared by the KG
    builder (node IDs) and the seed-refresh step in extraction/context.py,
    so both sides name the same definition identically.
    """
    seen: Dict[str, int] = {}
    out: List[Tuple[ast.AST, str]] = []

    def visit(scope: ast.AST, prefix: str) -> None:
        for d in iter_scope_defs(scope):
            path = f"{prefix}{d.name}"
            seen[path] = seen.get(path, 0) + 1
            qualname = path if seen[path] == 1 else f"{path}#{seen[path]}"
            out.append((d, qualname))
            visit(d, f"{qualname}.")

    visit(tree, "")
    return out


# ---------------------------------------------------------------------------
# AST unparsing
# ---------------------------------------------------------------------------

def _safe_unparse(node: ast.AST) -> Optional[str]:
    """Unparse an AST node to source string, returning None on failure.

    ast.unparse can fail on malformed or unusual AST nodes (e.g. from
    macro-generated code). Returning None rather than raising keeps the
    parse pipeline running.

    Used to capture:
    - Decorator expressions (e.g. @pytest.mark.parametrize("x", [1,2,3]))
    - Function signatures (parameter annotations and default values)
    - Exception expressions in raise statements (e.g. raise ValueError("bad input"))
    - Base classes in class definitions (e.g. class Foo(Bar, pkg.Base))
    """
    try:
        return ast.unparse(node)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Call-site extraction
# ---------------------------------------------------------------------------

def _extract_callee_name(call: ast.Call) -> Optional[str]:
    """Extract the function callee name from a Call node.

    TODO: This is a best-effort heuristic for simple call patterns. More complex patterns (e.g. dynamic calls, lambdas, calls via variables) are not handled, which may lead to missed edges but avoids false positives from unreliable inference.
    Handles:
        - Simple calls: foo() → 'foo'
        - Attribute calls: obj.method() → 'method'

    Does not handle:
        - Subscript calls: obj['key']() → None
        - Chained calls: foo()() → None
    """
    if isinstance(call.func, ast.Name):
        return call.func.id  # foo() → 'foo'
    if isinstance(call.func, ast.Attribute):
        return call.func.attr  # obj.method() → 'method'
    return None


def _outer_scope_exprs(def_node: Union[ast.FunctionDef, ast.AsyncFunctionDef]) -> List[ast.AST]:
    """The expressions of a def that run in the scope the def sits in, not in
    its own: decorators, default values and annotations."""
    args = def_node.args
    defaults = args.defaults + [d for d in args.kw_defaults if d is not None]
    params = args.posonlyargs + args.args + args.kwonlyargs
    params += [a for a in (args.vararg, args.kwarg) if a is not None]
    annotations = [a.annotation for a in params if a.annotation is not None]
    if def_node.returns is not None:
        annotations.append(def_node.returns)
    return def_node.decorator_list + defaults + annotations


def call_scope_chains(
    func_node: Union[ast.FunctionDef, ast.AsyncFunctionDef],
) -> Dict[int, List[ast.AST]]:
    """For every Call in a function, keyed by id(call): the functions around
    it, from `func_node` down to the innermost one. A call in `func_node`'s
    own body maps to [func_node]; one in a closure `inner` defined in it maps
    to [func_node, inner]. Walks the same way as iter_calls_by_scope."""
    return scope_chains(func_node, (ast.Call,))


def scope_chains(
    func_node: Union[ast.FunctionDef, ast.AsyncFunctionDef],
    kinds: Tuple[type, ...],
) -> Dict[int, List[ast.AST]]:
    """For every node of one of `kinds` in a function, keyed by id(node): the
    functions around it, as in call_scope_chains."""
    chains: Dict[int, List[ast.AST]] = {}
    stack: List[Tuple[ast.AST, Tuple[ast.AST, ...]]] = [(func_node, (func_node,))]
    while stack:
        node, chain = stack.pop()
        if isinstance(node, kinds):
            chains[id(node)] = list(chain)
        if node is not func_node and isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            kids = [(expr, chain) for expr in _outer_scope_exprs(node)]
            kids += [(stmt, chain + (node,)) for stmt in node.body]
        else:
            kids = [(child, chain) for child in ast.iter_child_nodes(node)]
        stack.extend(reversed(kids))
    return chains


def _lambda_params(args: ast.arguments) -> Set[str]:
    names = {a.arg for a in args.posonlyargs + args.args + args.kwonlyargs}
    names |= {a.arg for a in (args.vararg, args.kwarg) if a is not None}
    return names


def call_expression_scope_names(
    func_node: Union[ast.FunctionDef, ast.AsyncFunctionDef],
) -> Dict[int, FrozenSet[str]]:
    """For a Call inside a lambda or a comprehension, keyed by id(call): the
    names that expression's own scope binds (a lambda's parameters, a
    comprehension's loop variables). Calls with none are left out. A name in
    here is not the function's variable of the same name, whatever type that
    has. A comprehension's first iterable is evaluated in the enclosing scope,
    so it does not see the loop variables."""
    hidden: Dict[int, FrozenSet[str]] = {}
    none: FrozenSet[str] = frozenset()
    stack: List[Tuple[ast.AST, FrozenSet[str]]] = [(func_node, none)]
    while stack:
        node, names = stack.pop()
        if isinstance(node, ast.Call) and names:
            hidden[id(node)] = names
        if node is not func_node and isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            kids = [(expr, names) for expr in _outer_scope_exprs(node)]
            kids += [(stmt, none) for stmt in node.body]
        elif isinstance(node, ast.Lambda):
            defaults = node.args.defaults + [d for d in node.args.kw_defaults if d is not None]
            kids = [(d, names) for d in defaults]
            kids.append((node.body, names | _lambda_params(node.args)))
        elif isinstance(node, _COMPREHENSIONS):
            inner = names | {n.id for gen in node.generators for n in ast.walk(gen.target)
                             if isinstance(n, ast.Name)}
            kids = []
            for i, gen in enumerate(node.generators):
                kids.append((gen.iter, names if i == 0 else inner))
                kids += [(cond, inner) for cond in gen.ifs]
            kids += [(expr, inner) for expr in
                     (getattr(node, 'elt', None), getattr(node, 'key', None), getattr(node, 'value', None))
                     if expr is not None]
        else:
            kids = [(child, names) for child in ast.iter_child_nodes(node)]
        stack.extend(reversed(kids))
    return hidden


_COMPREHENSIONS = (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)


def _bound_names(
    func_node: Union[ast.FunctionDef, ast.AsyncFunctionDef],
) -> Tuple[Set[str], Set[str], Set[str]]:
    """The names a function binds in its own scope, as (bound, nonlocal, global).

    `bound` has its parameters, the names it assigns, loops over, or captures
    (`except ... as e`, `case ... as e`), and the names of the defs and classes
    it defines. A comprehension's loop variables are its own scope and are not
    included (a walrus inside one binds in the function, so it is). Imports are
    not included: ImportScopes handles those, in source order. A name declared
    `nonlocal` or `global` is not bound here, whatever is assigned to it.
    """
    args = func_node.args
    bound = {a.arg for a in args.posonlyargs + args.args + args.kwonlyargs}
    bound |= {a.arg for a in (args.vararg, args.kwarg) if a is not None}
    bound |= {d.name for d in iter_scope_defs(func_node)}
    nonlocals: Set[str] = set()
    globals_: Set[str] = set()

    def visit(node: ast.AST):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            bound.add(node.id)
        elif isinstance(node, ast.ExceptHandler) and node.name:
            bound.add(node.name)
        elif isinstance(node, (ast.MatchAs, ast.MatchStar)) and node.name:
            bound.add(node.name)
        elif isinstance(node, ast.MatchMapping) and node.rest:
            bound.add(node.rest)
        elif isinstance(node, ast.Nonlocal):
            nonlocals.update(node.names)
        elif isinstance(node, ast.Global):
            globals_.update(node.names)
        if isinstance(node, _COMPREHENSIONS):
            for gen in node.generators:
                visit(gen.iter)
                for cond in gen.ifs:
                    visit(cond)
            for expr in (getattr(node, 'elt', None), getattr(node, 'key', None), getattr(node, 'value', None)):
                if expr is not None:
                    visit(expr)
            return
        for child in ast.iter_child_nodes(node):
            if not isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                visit(child)

    for stmt in func_node.body:
        if not isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            visit(stmt)
    return bound - nonlocals - globals_, nonlocals, globals_


class LocalTypeScopes:
    """Looks up the class a receiver name holds at a call, the way Python
    resolves the name: in the innermost function around the call first, then
    outward (see call_scope_chains). Created once per outermost function; it
    caches what it computes per function.

    The first function in the chain that binds the name decides. If it knows
    the name's type (an annotated parameter, `x = SomeClass()`) that is the
    answer; if it binds the name without a known type there is none, and an
    outer function's type for the same name is not used, since that is a
    different variable. A name no function in the chain binds is a free
    variable of the outermost one, or a module level name.
    """

    def __init__(self, outer_func: ast.AST, outer_types: Dict[str, str]):
        self._types: Dict[int, Dict[str, str]] = {id(outer_func): outer_types}
        self._names: Dict[int, Tuple[Set[str], Set[str], Set[str]]] = {}
        self._imported: Dict[int, Set[str]] = {}

    def lookup(self, receiver: str, chain: List[ast.AST]) -> Tuple[bool, Optional[str]]:
        """(is the receiver a local of a function around the call, its type)."""
        for func in reversed(chain):
            names = self._names.get(id(func))
            if names is None:
                names = self._names[id(func)] = _bound_names(func)
            bound, nonlocals, globals_ = names
            if receiver in bound or receiver in globals_ or receiver in nonlocals:
                types = self._types.get(id(func))
                if types is None:
                    types = self._types[id(func)] = _collect_local_types(func)
                if receiver in bound:
                    return True, types.get(receiver)
                if receiver in types:
                    # `global x` / `nonlocal x` then `x = SomeClass()`: not a
                    # local, but this assignment says what the name holds here.
                    return False, types[receiver]
                if receiver in globals_:
                    return False, None  # a module level name: no enclosing function owns it
                # nonlocal: an enclosing function owns it, keep looking outward
            elif receiver in self._imports_of(func):
                # A function that imports the name has its own binding of it,
                # so an outer variable of that name is not what the call uses.
                # It is not a local with a type: ImportScopes resolves it.
                return False, None
        return False, None

    def _imports_of(self, func: ast.AST) -> Set[str]:
        imported = self._imported.get(id(func))
        if imported is None:
            imported = self._imported[id(func)] = set(scope_import_bindings(func))
        return imported


def iter_calls_by_scope(
    func_node: Union[ast.FunctionDef, ast.AsyncFunctionDef],
) -> Tuple[List[ast.Call], List[ast.Call]]:
    """Split the calls in a function into (own, in_closure), in source order.

    `own` are calls the function's own body makes. `in_closure` are calls
    inside a function nested in it: they only run if the closure is called,
    but they are still a dependency of the outer function. A nested
    function's decorators, default values and annotations are evaluated by
    the outer function, so they count as the outer function's own calls.
    Lambdas and comprehensions are part of the enclosing scope.
    """
    own: List[ast.Call] = []
    in_closure: List[ast.Call] = []
    stack: List[Tuple[ast.AST, bool]] = [(func_node, False)]
    while stack:
        node, nested = stack.pop()
        if isinstance(node, ast.Call):
            (in_closure if nested else own).append(node)
        if node is not func_node and isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            kids = [(expr, nested) for expr in _outer_scope_exprs(node)]
            kids += [(stmt, True) for stmt in node.body]
        else:
            kids = [(child, nested) for child in ast.iter_child_nodes(node)]
        stack.extend(reversed(kids))
    return own, in_closure


def _extract_call_receiver(call: ast.Call) -> Optional[str]:
    """Extract the receiver expression of an attribute call as a string.

    For obj.method() returns 'obj'; for json.loads() returns 'json';
    for self.foo.bar() returns 'self.foo'. Returns None for non-attribute
    calls (bare functions) or anything _safe_unparse can't render.

    The receiver lets pass-2 resolution distinguish json.loads from
    pickle.loads, and self.method from any other class's method.
    """
    if isinstance(call.func, ast.Attribute):
        return _safe_unparse(call.func.value)
    return None


def _annotated_param_types(
    func_node: Union[ast.FunctionDef, ast.AsyncFunctionDef],
) -> Dict[str, str]:
    """Map parameter names to their bare annotated type name, where unambiguous.

    Only direct Name/Attribute annotations are used (e.g. `x: Request` or
    `x: pkg.Request` -> 'Request'). Subscripted annotations like
    `Optional[Request]` or `List[Request]` are skipped: the parameter could
    be None or a container, so the "receiver is an instance of this type"
    assumption that _collect_local_types relies on doesn't hold.
    """
    types: Dict[str, str] = {}
    all_args = (
        func_node.args.posonlyargs
        + func_node.args.args
        + func_node.args.kwonlyargs
    )
    for arg in all_args:
        ann = arg.annotation
        if isinstance(ann, ast.Name):
            types[arg.arg] = ann.id
        elif isinstance(ann, ast.Attribute):
            types[arg.arg] = ann.attr
    return types


def _extract_property_accesses(
    func_node: Union[ast.FunctionDef, ast.AsyncFunctionDef],
) -> List[Tuple[str, Optional[str]]]:
    """Find genuine (non-call) attribute reads: `obj.attr`, not `obj.attr()`.

    A @property-decorated method is invoked as `obj.attr`, never
    `obj.attr()` -- it never appears as an ast.Call node, so
    _extract_callee_name/_emit_call_edges have zero signal for it. This
    walks Attribute nodes directly and excludes:
      - Attribute nodes that are the .func of an enclosing Call (already
        covered by 'calls' edge extraction -- obj.method() is a call, not
        a property access)
      - Attribute nodes in Store/Del context (assignment targets, e.g.
        `obj.attr = x` -- a write, not a read)
      - Nested function scopes (their accesses belong to that scope)

    Returns a deduplicated list of (attr_name, receiver) pairs, e.g.
    `self.headers` -> ('headers', 'self'), `session.cookies` ->
    ('cookies', 'session'). receiver is None if _safe_unparse fails.
    """
    accesses: List[Tuple[str, Optional[str]]] = []
    seen: Set[Tuple[str, Optional[str]]] = set()
    call_funcs: Set[int] = set()

    def _collect_call_funcs(n: ast.AST):
        for child in ast.walk(n):
            if isinstance(child, ast.Call) and isinstance(child.func, ast.Attribute):
                call_funcs.add(id(child.func))

    _collect_call_funcs(func_node)

    def _walk_no_nested(n: ast.AST):
        for child in ast.iter_child_nodes(n):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue  # nested fn -- its accesses belong to that scope
            if (isinstance(child, ast.Attribute)
                    and isinstance(child.ctx, ast.Load)
                    and id(child) not in call_funcs):
                receiver = _safe_unparse(child.value)
                key = (child.attr, receiver)
                if key not in seen:
                    seen.add(key)
                    accesses.append(key)
            _walk_no_nested(child)

    _walk_no_nested(func_node)
    return accesses


# Common overloadable binary operators -> the dunder method Python
# dispatches to. Right-operand fallback (__ror__ etc, when the left
# side returns NotImplemented) and in-place variants (__iadd__ etc)
# are out of scope for a first cut (kg_construction#122).
_BINOP_DUNDER_METHODS = {
    ast.Add: '__add__', ast.Sub: '__sub__', ast.Mult: '__mul__',
    ast.Div: '__truediv__', ast.FloorDiv: '__floordiv__', ast.Mod: '__mod__',
    ast.Pow: '__pow__', ast.LShift: '__lshift__', ast.RShift: '__rshift__',
    ast.BitOr: '__or__', ast.BitXor: '__xor__', ast.BitAnd: '__and__',
    ast.MatMult: '__matmul__',
}

_COMPARE_DUNDER_METHODS = {
    ast.Eq: '__eq__', ast.NotEq: '__ne__', ast.Lt: '__lt__',
    ast.LtE: '__le__', ast.Gt: '__gt__', ast.GtE: '__ge__',
}


def _extract_operator_dispatches(
    func_node: Union[ast.FunctionDef, ast.AsyncFunctionDef],
) -> List[Tuple[str, str]]:
    """Find operator sites whose left operand is a bare name (kg_construction#122).

    `a | b` never appears as an ast.Call, so _emit_call_edges has no
    signal for it at all -- Python's own dispatch routes it to
    `a.__or__(b)`. Only the left operand is handled (no __ror__
    fallback), and only when it's a bare Name -- the same shape
    _extract_call_receiver already resolves for `x.method()`, so the
    existing local_types/class_hint machinery can resolve it without
    any new type inference. Chained comparisons (`a < b < c`) are
    skipped, only the simple a OP b shape is handled.

    Returns a deduplicated list of (dunder_method, receiver) pairs,
    e.g. `self | x` -> ('__or__', 'self').
    """
    dispatches: List[Tuple[str, str]] = []
    seen: Set[Tuple[str, str]] = set()

    def _walk_no_nested(n: ast.AST):
        for child in ast.iter_child_nodes(n):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue  # nested fn -- its operator sites belong to that scope

            dunder = None
            left = None
            if isinstance(child, ast.BinOp):
                dunder = _BINOP_DUNDER_METHODS.get(type(child.op))
                left = child.left
            elif isinstance(child, ast.Compare) and len(child.ops) == 1:
                dunder = _COMPARE_DUNDER_METHODS.get(type(child.ops[0]))
                left = child.left

            if dunder and isinstance(left, ast.Name):
                key = (dunder, left.id)
                if key not in seen:
                    seen.add(key)
                    dispatches.append(key)

            _walk_no_nested(child)

    _walk_no_nested(func_node)
    return dispatches


def _collect_local_types(
    func_node: Union[ast.FunctionDef, ast.AsyncFunctionDef],
    before_line: Optional[int] = None,
) -> Dict[str, str]:
    """Map local variable names to inferred class names from constructor calls
    and type-annotated parameters.

    Seeds the map from parameter annotations first (e.g. `def f(self, x: Request)`
    -> {'x': 'Request'}), then walks assignments of the form `x = SomeClass(...)`
    and records {x: SomeClass}, overwriting the annotation-derived entry if the
    same name is reassigned to a constructor call. Heuristics for constructor
    calls: only direct calls (capitalized name or attribute call whose final
    part is capitalized). Nested function bodies are skipped for assignments —
    their locals belong to that scope — but a nested function's own parameters
    are not in scope here either way.

    Args:
        before_line: If given, only assignments whose target line is strictly
            before this 1-indexed line number are considered -- callers that
            need "what type did x hold at a specific call site" (rather than
            "what's the last type x is ever assigned across the whole
            function") must pass the call site's line, since a later
            reassignment elsewhere in the function is otherwise
            indistinguishable from the type in effect at an earlier site.

    Used to resolve attribute calls like `x.save()` to `SomeClass.save` when x's
    type is known from a visible constructor call or parameter annotation.
    """
    types: Dict[str, str] = _annotated_param_types(func_node)

    def _walk(n: ast.AST):
        for child in ast.iter_child_nodes(n):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue  # nested fn — its locals belong to that scope
            if isinstance(child, ast.Assign) and isinstance(child.value, ast.Call):
                if before_line is not None and child.lineno >= before_line:
                    _walk(child)
                    continue
                call = child.value
                cls_name: Optional[str] = None
                if isinstance(call.func, ast.Name) and call.func.id and call.func.id[0].isupper():
                    cls_name = call.func.id
                elif isinstance(call.func, ast.Attribute) and call.func.attr and call.func.attr[0].isupper():
                    cls_name = call.func.attr
                if cls_name is not None:
                    for target in child.targets:
                        if isinstance(target, ast.Name):
                            types[target.id] = cls_name
            _walk(child)

    _walk(func_node)
    return types


# ---------------------------------------------------------------------------
# Function/method metadata
# ---------------------------------------------------------------------------

def _get_docstring(node: ast.AST) -> Optional[str]:
    """Extract the docstring from a function, class, or module node.

    Returns the string value of the first expression statement if it is
    a string constant, otherwise None.
    """
    body = getattr(node, 'body', [])
    # Docstring is typically the first statement in the body and must be a string constant.
    if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
        val = body[0].value.value
        if isinstance(val, str):
            return val.strip()
    return None


def _get_decorators(node: ast.AST) -> List[str]:
    """Return the list of decorator expressions as unparsed strings.

    Handles plain names (@staticmethod), attribute access (@pytest.mark.skip),
    and decorator calls (@pytest.mark.parametrize("x", [1,2,3])).
    """
    return [
        _safe_unparse(d) or ''
        for d in getattr(node, 'decorator_list', [])
        if _safe_unparse(d)
    ]


def _get_bare_decorator_names(node: ast.AST) -> List[str]:
    """Bare names of unqualified decorators only (kg_construction#113).

    A dotted/attribute decorator (@pytest.mark.skip, @app.route) isn't a
    plain function/class defined in this repo the same way @my_decorator
    is -- only bare Name and bare Call(func=Name(...)) forms are
    resolvable against the repo's own function/class index. Whether the
    resolved name turns out to be stdlib (@staticmethod) or a real
    in-repo decorator is left to the index lookup itself: a stdlib name
    simply won't match anything and the edge gets dropped, same as any
    other unresolved edge in this codebase.
    """
    names = []
    for dec in getattr(node, 'decorator_list', []):
        target = dec.func if isinstance(dec, ast.Call) else dec
        if isinstance(target, ast.Name):
            names.append(target.id)
    return names


def _get_signature(node: Union[ast.FunctionDef, ast.AsyncFunctionDef]) -> Dict:
    """Extract the full parameter signature of a function or method.

    Returns a dict with:
        params: list of {name, annotation?, default?} dicts in declaration order.
            - Positional-only and regular args are listed first.
            - Keyword-only args follow.
            - *args and **kwargs appear last, prefixed with * and **.
        returns: unparsed return annotation string, if present.

    Defaults are right-aligned against the args list per Python semantics
    (i.e. def f(a, b=1) has defaults=[1] aligned to b, not a).
    """
    args = node.args

    # defaults is right-aligned: pad left with None for args that have no default
    all_args = args.posonlyargs + args.args

    # Align defaults to the end of all_args: if there are 2 args and 1 default, we want [None, default] not [default]
    defaults = [None] * (len(all_args) - len(args.defaults)) + list(args.defaults)

    params = []
    for i, arg in enumerate(all_args):
        param: Dict = {'name': arg.arg}
        if arg.annotation:
            param['annotation'] = _safe_unparse(arg.annotation)
        if defaults[i] is not None:
            param['default'] = _safe_unparse(defaults[i])
        params.append(param)

    # kw_defaults is parallel to kwonlyargs, entries are None where there's no default
    for i, arg in enumerate(args.kwonlyargs):
        param = {'name': arg.arg}
        if arg.annotation:
            param['annotation'] = _safe_unparse(arg.annotation)
        kw_default = args.kw_defaults[i] if i < len(args.kw_defaults) else None  # safety check in case of malformed AST where kw_defaults is shorter than kwonlyargs
        if kw_default is not None:
            param['default'] = _safe_unparse(kw_default)
        params.append(param)

    if args.vararg:  # *args
        param = {'name': f"*{args.vararg.arg}"}
        if args.vararg.annotation:
            param['annotation'] = _safe_unparse(args.vararg.annotation)
        params.append(param)

    if args.kwarg:  # **kwargs
        param = {'name': f"**{args.kwarg.arg}"}
        if args.kwarg.annotation:
            param['annotation'] = _safe_unparse(args.kwarg.annotation)
        params.append(param)

    result: Dict = {'params': params}
    if node.returns:
        result['returns'] = _safe_unparse(node.returns)
    return result


def _get_signature_and_source_text(
    func_node: Union[ast.FunctionDef, ast.AsyncFunctionDef],
    source_lines: Optional[List[str]],
) -> Tuple[str, str]:
    """Slice a function's own `def` line(s) and full source text out of its
    enclosing file, via the AST node's line range.

    Both are LLM-facing convenience fields (LLMSerializer renders them
    directly into a test-generation prompt) -- distinct from `params`,
    which is structured data for programmatic use. Prior to this, neither
    was ever populated, so the KG-augmented arm's prompt showed a
    docstring and a param-name list but never the actual code being
    tested (see kg-test-generation#51/#25/#28 -- several test-quality
    bugs trace back to the model never seeing the function's real body).

    Returns:
        (signature, source_code) -- both "" if source_lines isn't given,
        or if func_node.body is empty (should not happen for valid AST,
        but avoids an index error on malformed input rather than raising).
    """
    if not source_lines or not func_node.body:
        return "", ""

    source_code = "".join(source_lines[func_node.lineno - 1 : func_node.end_lineno])

    # The signature is everything from the `def`/`async def` line up to
    # (not including) the first body statement -- covers multi-line
    # signatures (wrapped params) without pulling in the function body.
    first_body_lineno = func_node.body[0].lineno
    signature_end = first_body_lineno - 1 if first_body_lineno > func_node.lineno else func_node.lineno
    signature = "".join(source_lines[func_node.lineno - 1 : signature_end]).rstrip()
    return signature, source_code


def _get_exceptions(node: Union[ast.FunctionDef, ast.AsyncFunctionDef]) -> Dict:
    """Extract exception types raised and caught within a function body.

    Returns:
        raises: deduplicated list of unparsed raise expressions,
                e.g. ['ValueError("bad input")', 'KeyError(key)']
        catches: deduplicated list of unparsed except handler types,
                 e.g. ['KeyError', 'TypeError']

    Walks the entire function body including nested calls, so re-raises
    and chained handlers are all captured.
    """
    raises, catches = [], []
    for child in ast.walk(node):
        if isinstance(child, ast.Raise) and child.exc is not None:
            unparsed = _safe_unparse(child.exc)
            if unparsed:
                raises.append(unparsed)
        elif isinstance(child, ast.ExceptHandler) and child.type is not None:
            unparsed = _safe_unparse(child.type)
            if unparsed:
                catches.append(unparsed)
    # sorted(), not list(set(...)): Python randomizes set iteration
    # order for strings per-process by default (PYTHONHASHSEED unset),
    # so two separate builds of the same function could otherwise
    # report 'raises'/'catches' in a different order. This feeds
    # exceptions in the rendered prompt directly (SEED_BLOCK_TEMPLATE's
    # "Declared exceptions" field), the same class of bug as #145's
    # BFS visited-node-order fix, just a different call site.
    return {'raises': sorted(set(raises)), 'catches': sorted(set(catches))}


def _extract_exception_class_name(expr: ast.AST) -> Optional[str]:
    """Bare class name from a single raise/except expression, if resolvable.

    Handles `raise Foo`, `raise Foo(...)`, `except Foo`. Attribute-based
    forms (`raise mod.Foo(...)`, `raise self.exc(...)`) are left
    unresolved here -- module.Class needs import resolution, and
    self.attr is an instance attribute, not a class name at all.
    """
    target = expr.func if isinstance(expr, ast.Call) else expr
    if isinstance(target, ast.Name):
        return target.id
    return None


def _get_exception_class_names(node: Union[ast.FunctionDef, ast.AsyncFunctionDef]) -> Dict:
    """Bare, resolvable class names raised/caught in a function body.

    Companion to _get_exceptions: that returns full unparsed expressions
    for display; this returns just the class names usable to resolve a
    real edge to the exception class's own node (kg_construction#112).
    A tuple except handler (`except (A, B):`) contributes every element.
    """
    raises, catches = set(), set()
    for child in ast.walk(node):
        if isinstance(child, ast.Raise) and child.exc is not None:
            name = _extract_exception_class_name(child.exc)
            if name:
                raises.add(name)
        elif isinstance(child, ast.ExceptHandler) and child.type is not None:
            exc_types = child.type.elts if isinstance(child.type, ast.Tuple) else [child.type]
            for exc_type in exc_types:
                name = _extract_exception_class_name(exc_type)
                if name:
                    catches.add(name)
    return {'raises': sorted(raises), 'catches': sorted(catches)}


def _extract_conditions(node: Union[ast.FunctionDef, ast.AsyncFunctionDef]) -> List[Dict]:
    """Extract boundary conditions from if, while, and assert statements.

    Returns a list of {type, condition, lineno} dicts representing the conditions
    that guard behavior in the function. Used for test generation to identify
    edge cases and boundary values to test.

    Does NOT descend into nested function scopes to avoid capturing conditions
    from helper functions defined within the target function.

    Examples:
        if x < 0: ... → {type: 'if', condition: 'x < 0', lineno: 5}
        while y > 100: ... → {type: 'while', condition: 'y > 100', lineno: 10}
        assert z != 0 → {type: 'assert', condition: 'z != 0', lineno: 15}
    """
    conditions: List[Dict] = []
    seen: Set[Tuple[str, int]] = set()

    def _walk_no_nested(n: ast.AST) -> None:
        """Recursively extract conditions, stopping at nested function definitions."""
        for child in ast.iter_child_nodes(n):
            # Skip nested function scopes entirely
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue

            if isinstance(child, ast.If):
                unparsed = _safe_unparse(child.test)
                if unparsed:
                    key = ('if', child.lineno)
                    if key not in seen:
                        seen.add(key)
                        conditions.append({'type': 'if', 'condition': unparsed, 'lineno': child.lineno})

            elif isinstance(child, ast.While):
                unparsed = _safe_unparse(child.test)
                if unparsed:
                    key = ('while', child.lineno)
                    if key not in seen:
                        seen.add(key)
                        conditions.append({'type': 'while', 'condition': unparsed, 'lineno': child.lineno})

            elif isinstance(child, ast.Assert):
                unparsed = _safe_unparse(child.test)
                if unparsed:
                    key = ('assert', child.lineno)
                    if key not in seen:
                        seen.add(key)
                        conditions.append({'type': 'assert', 'condition': unparsed, 'lineno': child.lineno})

            # Recurse into this child (will skip nested functions)
            _walk_no_nested(child)

    _walk_no_nested(node)
    return conditions


def _extract_data_flows(node: Union[ast.FunctionDef, ast.AsyncFunctionDef]) -> Dict:
    """Extract data flow information from a function body.

    Identifies how data flows through the function:
    - Return values flowing to callers
    - Parameter usage patterns
    - Attribute mutations on self

    Returns a dict with keys:
        returns: List of unparsed return value expressions
        mutates_attributes: {attr_name: [list of value expressions assigned to it]}
        parameter_usage: {param_name: [line numbers where it appears]}

    Example:
        def send(self, url, timeout=30):
            if timeout < 0:
                raise ValueError()
            response = self._request(url)
            self.cache[url] = response
            return response

        Returns:
        {
            'returns': ['response'],
            'mutates_attributes': {'cache': ['self._request(url)']},
            'parameter_usage': {'url': [4, 5], 'timeout': [3]}
        }
    """
    flows = {
        'returns': [],
        'mutates_attributes': {},
        'parameter_usage': {},
    }

    # Get parameter names for later lookup
    param_names = set()
    for arg in node.args.args + node.args.posonlyargs + node.args.kwonlyargs:
        param_names.add(arg.arg)
    if node.args.vararg:
        param_names.add(node.args.vararg.arg)
    if node.args.kwarg:
        param_names.add(node.args.kwarg.arg)

    # Walk function body (skip nested function defs for mutations, but track param usage in them)
    def _walk_no_nested(n: ast.AST, allow_param_in_nested: bool = False) -> None:
        """Recursively extract data flows.

        Args:
            n: AST node to walk
            allow_param_in_nested: If True, we're inside a nested function and should track param usage
        """
        for child in ast.iter_child_nodes(n):
            # For nested functions, extract param usage but skip mutations/returns
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                # Recursively walk nested function only for parameter usage (closures)
                _walk_no_nested(child, allow_param_in_nested=True)
                continue

            # Track return values (only in direct scope, not nested functions)
            if isinstance(child, ast.Return) and child.value and not allow_param_in_nested:
                unparsed = _safe_unparse(child.value)
                if unparsed:
                    flows['returns'].append(unparsed)

            # Track attribute mutations (only in direct scope, not nested
            # functions), through every form _get_attribute_accesses counts
            # as a write: self.x = value, self.x[key] = value, augmented and
            # annotated assignment, unpacking, and for/with targets.
            elif (isinstance(child, (ast.Assign, ast.AnnAssign, ast.AugAssign,
                                     ast.For, ast.AsyncFor, ast.With, ast.AsyncWith))
                  and not allow_param_in_nested):
                value = getattr(child, 'value', None)
                value_unparsed = _safe_unparse(value) if value is not None else None
                for attr_name in _written_self_attrs(child, include_subscript=True):
                    mutated = flows['mutates_attributes'].setdefault(attr_name, [])
                    if value_unparsed:
                        mutated.append(value_unparsed)

            # Track parameter usage (Name nodes in Load context)
            elif isinstance(child, ast.Name) and isinstance(child.ctx, ast.Load):
                if child.id in param_names:
                    param = child.id
                    if param not in flows['parameter_usage']:
                        flows['parameter_usage'][param] = []
                    flows['parameter_usage'][param].append(child.lineno)

            # Recurse, keeping the flag: inside a closure, nothing deeper
            # (an `if` body, a loop) may count as the outer function's own
            # returns or mutations either.
            _walk_no_nested(child, allow_param_in_nested)

    _walk_no_nested(node)

    # Deduplicate and sort parameter usage lines
    for param in flows['parameter_usage']:
        flows['parameter_usage'][param] = sorted(list(set(flows['parameter_usage'][param])))

    # Deduplicate return values and mutations. sorted(), not
    # list(set(...)), for the same reason as parameter_usage above and
    # _extract_exceptions' raises/catches: Python randomizes set
    # iteration order for strings per-process by default.
    flows['returns'] = sorted(set(flows['returns']))
    for attr in flows['mutates_attributes']:
        flows['mutates_attributes'][attr] = sorted(set(flows['mutates_attributes'][attr]))

    return flows


def _count_branches(node: Union[ast.FunctionDef, ast.AsyncFunctionDef]) -> int:
    """Count control flow branches (if/for/while) in this function only.

    Includes all branches in the function body
    """
    count = 0

    def _walk(n):
        nonlocal count
        for child in ast.iter_child_nodes(n):
            if child is not node and isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            # Count if, for, and while statements as branches. Other control flow constructs (try/except, with) could be added in the future if desired.
            if isinstance(child, (ast.If, ast.For, ast.While)):
                count += 1
            _walk(child)

    _walk(node)
    return count


def _get_assert_patterns(node: Union[ast.FunctionDef, ast.AsyncFunctionDef]) -> List[str]:
    """Extract assertion patterns from a test function.

    Captures two forms:
        - assert statements: `assert x == y` → 'x == y'
        - unittest-style method calls: `self.assertEqual(...)` → full call string

    These patterns are used by the test generator to understand what
    properties existing tests verify, informing generation of new assertions.
    """
    patterns = []
    for child in ast.walk(node):
        if isinstance(child, ast.Assert):
            unparsed = _safe_unparse(child.test)
            if unparsed:
                patterns.append(unparsed)
        elif (isinstance(child, ast.Call) and
              isinstance(child.func, ast.Attribute) and
              child.func.attr.startswith('assert')):
            unparsed = _safe_unparse(child)
            if unparsed:
                patterns.append(unparsed)
    return patterns


# ---------------------------------------------------------------------------
# Class metadata
# ---------------------------------------------------------------------------

def _get_base_names(node: ast.ClassDef) -> List[str]:
    """Return the unparsed base class expressions for a class definition.

    e.g. class Foo(Bar, Mixin) → ['Bar', 'Mixin']
         class Foo(pkg.Base)   → ['pkg.Base']
    """
    return [_safe_unparse(base) for base in node.bases if _safe_unparse(base)]


def _walk_own_scope(node: ast.AST):
    """Yield `node` and its descendants, without entering the body of a nested
    function definition: its statements belong to its own scope. Its
    decorators, default values and annotations run in this scope, so they are
    walked (the same boundary as iter_calls_by_scope)."""
    yield node
    for child in ast.iter_child_nodes(node):
        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for expr in _outer_scope_exprs(child):
                yield from _walk_own_scope(expr)
            continue
        yield from _walk_own_scope(child)


def _self_attr_names(target: ast.AST, include_subscript: bool) -> List[str]:
    """Names of `self.<name>` attributes an assignment target writes.

    Handles `self.x`, tuple and list unpacking (including starred
    elements), and, if include_subscript, `self.x[key]` (a write to the
    container held in x).
    """
    if isinstance(target, ast.Attribute):
        if isinstance(target.value, ast.Name) and target.value.id == 'self':
            return [target.attr]
        return []
    if isinstance(target, ast.Subscript) and include_subscript:
        value = target.value
        if (isinstance(value, ast.Attribute) and isinstance(value.value, ast.Name)
                and value.value.id == 'self'):
            return [value.attr]
        return []
    if isinstance(target, (ast.Tuple, ast.List)):
        return [name for elt in target.elts for name in _self_attr_names(elt, include_subscript)]
    if isinstance(target, ast.Starred):
        return _self_attr_names(target.value, include_subscript)
    return []


def _written_self_attrs(stmt: ast.AST, include_subscript: bool) -> List[str]:
    """`self.<name>` attributes written by one statement: plain, annotated
    and augmented assignment, `for` targets, and `with ... as` targets.
    Mutating method calls such as `self.items.append(x)` are not writes."""
    if isinstance(stmt, ast.Assign):
        targets = stmt.targets
    elif isinstance(stmt, (ast.AnnAssign, ast.AugAssign)):
        targets = [stmt.target]
    elif isinstance(stmt, (ast.For, ast.AsyncFor)):
        targets = [stmt.target]
    elif isinstance(stmt, (ast.With, ast.AsyncWith)):
        targets = [item.optional_vars for item in stmt.items if item.optional_vars is not None]
    else:
        return []
    return [name for t in targets for name in _self_attr_names(t, include_subscript)]


def _get_class_attributes(node: ast.ClassDef) -> List[str]:
    """Extract instance variable names assigned in __init__.

    Covers plain, annotated and augmented assignment, tuple and list
    unpacking, and `for`/`with` targets (self.x = ..., self.a, self.b = ...,
    self.n += 1). Returns names in assignment order, deduplicated.

    Only __init__ is scanned, and only its own scope: attributes set in
    other methods, or inside a function nested in __init__, are not
    included, as they may not always exist on the instance.
    """
    attrs = []
    for child in iter_scope_defs(node):
        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)) and child.name == '__init__':
            for stmt in _walk_own_scope(child):
                attrs.extend(_written_self_attrs(stmt, include_subscript=False))
    # dict.fromkeys preserves insertion order while deduplicating
    return list(dict.fromkeys(attrs))


def _get_instantiated_classes_in_class(class_node: ast.ClassDef) -> List[str]:
    """Collect unique class names instantiated across all methods of a class.

    Aggregates _get_instantiated_classes over every method body. Used to emit
    class→uses→class edges at the class level: if any method of class A
    instantiates class B, A is considered a direct dependency of B.

    Args:
        class_node: The ClassDef AST node to inspect.

    Returns:
        Deduplicated list of instantiated class name strings (PEP 8 uppercase
        convention). Names are unresolved — second-pass resolution maps them
        to class node IDs via class_label_to_ids.
    """
    seen: List[str] = []
    for child in iter_scope_defs(class_node):
        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for name in _get_instantiated_classes(child):
                if name not in seen:
                    seen.append(name)
    return seen


# ---------------------------------------------------------------------------
# Function-body analysis
# ---------------------------------------------------------------------------

def _get_attribute_accesses(
    func_node: Union[ast.FunctionDef, ast.AsyncFunctionDef],
    class_name: Optional[str] = None,
) -> Tuple[List[str], List[str]]:
    """Extract self.x reads and self.x writes from a method body.

    Reads are any `self.attr` access that is not a write. Writes are every
    form that assigns to `self.attr` or `self.attr[key]`: plain, annotated
    and augmented assignment, tuple and list unpacking, and `for`/`with`
    targets. Mutating method calls (`self.items.append(x)`) are not writes.
    Excludes nested function scopes to avoid false positives from closures.

    Args:
        func_node: The method AST node.
        class_name: Used only for scoping, not currently used but kept for
                    future qualified-attribute support.

    Returns:
        (reads, writes): deduplicated lists of attribute name strings.
    """
    reads: List[str] = []
    writes: List[str] = []
    written: Set[str] = set()

    for child in _walk_own_scope(func_node):
        names = _written_self_attrs(child, include_subscript=True)
        if names:
            writes.extend(names)
            written.update(names)
            if isinstance(child, ast.AugAssign):
                reads.extend(names)  # `self.x += 1` reads the old value first
        elif isinstance(child, ast.Attribute):
            if (isinstance(child.value, ast.Name) and
                    child.value.id == 'self' and
                    child.attr not in written):
                reads.append(child.attr)

    return list(dict.fromkeys(reads)), list(dict.fromkeys(writes))


def _get_used_imports(
    func_node: Union[ast.FunctionDef, ast.AsyncFunctionDef],
    scopes: "ImportScopes",
) -> List[str]:
    """Find which imported names are actually referenced inside a function.

    Each Name is resolved through the import that has run by the time it is
    reached (see ImportScopes), the same way calls are, so a name bound by
    two imports in turn yields both: `client.run(); import b as client;
    client.run()` uses the outer `client` and then b (#200). Returns the
    fully-qualified import paths, which tells the test generator exactly
    what to mock. Nested functions are included on purpose: an import used
    only inside a closure still has to be mocked to test the outer function.

    Args:
        func_node: The function AST node.
        scopes: ImportScopes over the imports the function sees from outside.

    Returns:
        List of fully-qualified import strings used in this function,
        e.g. ['requests.auth.HTTPBasicAuth', 'os.path'].
    """
    used: List[str] = []
    chains = scope_chains(func_node, (ast.Name,))
    # A function's own `if TYPE_CHECKING:` imports are not runtime bindings,
    # so ImportScopes does not see them, but a name used only for an
    # annotation still depends on that module. Used when no runtime import
    # binds the name; computed once per function.
    type_only: Dict[int, Dict[str, str]] = {}
    for child in ast.walk(func_node):
        if not isinstance(child, ast.Name):
            continue
        chain = chains.get(id(child), [func_node])
        found = scopes.lookup_all(child.id, child, chain)
        if not found:
            for scope in reversed(chain):
                if id(scope) not in type_only:
                    type_only[id(scope)] = _split_scope_imports(scope)[1]
                if child.id in type_only[id(scope)]:
                    found = [type_only[id(scope)][child.id]]
                    break
        for qualified in found:
            if qualified not in used:
                used.append(qualified)
    return used


def _get_instantiated_classes(
    func_node: Union[ast.FunctionDef, ast.AsyncFunctionDef],
) -> List[str]:
    """Find class names instantiated inside a function (e.g. Foo(...) calls).

    Detects patterns like `x = SomeClass(...)` by finding Call nodes where
    the callee is a Name starting with an uppercase letter (PEP 8 convention).
    Used to build `uses` edges from a class to its dependencies. Nested
    functions are included on purpose: a class instantiated inside a closure
    is still a dependency of the outer function.

    Returns:
        List of class name strings instantiated in this function.
    """
    classes: List[str] = []
    for child in ast.walk(func_node):
        if isinstance(child, ast.Call):
            if isinstance(child.func, ast.Name) and child.func.id[0].isupper():
                if child.func.id not in classes:
                    classes.append(child.func.id)
            elif isinstance(child.func, ast.Attribute) and child.func.attr[0].isupper():
                if child.func.attr not in classes:
                    classes.append(child.func.attr)
    return classes


def _get_factory_call_sites(
    func_node: Union[ast.FunctionDef, ast.AsyncFunctionDef],
) -> List[Tuple[int, int, Optional[str]]]:
    """Find variable assignments from lowercase-named calls (candidate factories).

    Detects patterns like `x = some_call(...)`, `self.x = some_call(...)`,
    and `other.x = some_call(...)` where the callee name does NOT start with
    an uppercase letter — the exact complement of the heuristic in
    _get_instantiated_classes(), which only catches uppercase (PEP 8 class
    convention) callees. A lowercase callee returning a class instance (e.g.
    `self.session = requests.session()`) is invisible to that heuristic;
    this function instead records the assignment's source location so an
    external type checker (see kg/type_inference.py) can be asked what type
    the right-hand side actually evaluates to.

    Only simple `Name` targets (`x = ...`) and single-level attribute targets
    (`receiver.x = ...`, where receiver is itself a simple `Name`) are
    recorded; tuple targets and attribute chains deeper than one level are
    skipped since they don't reduce to a single (line, col) an editor-style
    hover query resolves cleanly.

    Returns:
        List of (lineno, col_offset, receiver) tuples, one per candidate
        assignment target, 0-indexed line/col ready for an LSP-style
        position query (ast reports 1-indexed lines, converted here).
        `receiver` is None for a bare `Name` target (`x = ...`) or a
        `self.x = ...` target (the enclosing class is already known from
        context); for any other `receiver.x = ...` target, `receiver` is
        that name, so the caller can resolve the receiver's own type (e.g.
        via _collect_local_types) to know which class the resulting `uses`
        edge should originate from. For attribute targets the position
        points at the attribute name itself, not the receiver.
    """
    sites: List[Tuple[int, int, Optional[str]]] = []

    def _walk(n: ast.AST):
        for child in ast.iter_child_nodes(n):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue  # nested fn — its locals belong to that scope
            if isinstance(child, ast.Assign) and isinstance(child.value, ast.Call):
                call = child.value
                callee_name: Optional[str] = None
                if isinstance(call.func, ast.Name):
                    callee_name = call.func.id
                elif isinstance(call.func, ast.Attribute):
                    callee_name = call.func.attr
                if callee_name and not callee_name[0].isupper():
                    for target in child.targets:
                        if isinstance(target, ast.Name):
                            sites.append((target.lineno - 1, target.col_offset, None))
                        elif (
                            isinstance(target, ast.Attribute)
                            and isinstance(target.value, ast.Name)
                        ):
                            # Position the query at the attribute name itself
                            # (e.g. the 's' in 'self.s'), not at the receiver.
                            attr_col = target.end_col_offset - len(target.attr)
                            receiver = None if target.value.id == 'self' else target.value.id
                            sites.append((target.lineno - 1, attr_col, receiver))
            _walk(child)

    _walk(func_node)
    return sites


# ---------------------------------------------------------------------------
# Aggregators
# ---------------------------------------------------------------------------

def _get_annotation_type_names(node: Union[ast.FunctionDef, ast.AsyncFunctionDef]) -> List[str]:
    """Extract bare type names from all parameter annotations and the return annotation.

    Walks annotation AST nodes to find Name and Attribute nodes, extracting just the
    final component (e.g., pkg.MyClass → 'MyClass', Optional[MyClass] → 'Optional', 'MyClass').
    Used to identify external type dependencies from type hints.
    """
    names: List[str] = []

    def _collect(ann: ast.AST):
        if isinstance(ann, ast.Name):
            names.append(ann.id)
        elif isinstance(ann, ast.Attribute):
            names.append(ann.attr)
        for child in ast.iter_child_nodes(ann):
            _collect(child)

    for arg in node.args.posonlyargs + node.args.args + node.args.kwonlyargs:
        if arg.annotation:
            _collect(arg.annotation)
    if node.returns:
        _collect(node.returns)

    return list(dict.fromkeys(names))


def _build_func_metadata(
    func_node: Union[ast.FunctionDef, ast.AsyncFunctionDef],
    rel_path: str,
    repo: str,
    parent_class: Optional[str] = None,
    parent_function: Optional[str] = None,
    import_map: Optional[Dict[str, str]] = None,
    source_lines: Optional[List[str]] = None,
) -> Dict:
    """Build the full metadata dict for a function or method node.

    Centralises metadata construction to avoid duplication between the
    top-level function, class method, and nested-function branches of
    _parse_file.

    Args:
        func_node: The AST function node.
        rel_path: Relative file path within the repo.
        repo: Repository name (e.g. 'psf/requests').
        parent_class: Class name if this is a method, else None.
        parent_function: Enclosing function's name if this is a nested
                         function/closure, else None (kg_construction#74).
                         Mutually exclusive with parent_class in practice --
                         a nested function's own enclosing scope is either
                         a class method or a function, not both.
        import_map: {local_name: fully_qualified_name} dict for resolving
                    external type dependencies. Defaults to {}.
        source_lines: The enclosing file's source, split via
                      split_source_lines, used to slice out this
                      function's exact source text and its `def` line(s) for
                      the 'source_code'/'signature' metadata fields. If not
                      given (e.g. a caller with no file text handy), both
                      fields are left empty rather than failing the build --
                      LLMSerializer already treats them as optional.

    Returns:
        Dict with keys: filepath, repo, lineno, params, returns, signature,
        source_code, decorators, docstring, raises, catches, is_async,
        branches, assert_patterns, external_deps, side_effects, and
        optionally 'class' (parent_class) or 'parent_function'
        (parent_function), whichever was given.
    """
    sig = _get_signature(func_node)
    exc = _get_exceptions(func_node)
    _import_map = import_map or {}
    signature_str, source_code = _get_signature_and_source_text(func_node, source_lines)

    # external_deps: annotation types + instantiated classes that appear in import_map
    annotation_types = _get_annotation_type_names(func_node)
    instantiated = _get_instantiated_classes(func_node)
    all_type_names = list(dict.fromkeys(annotation_types + instantiated))
    external_deps = [
        _import_map[name] for name in all_type_names if name in _import_map
    ]

    # side_effects: attribute names written via self.*
    _, writes = _get_attribute_accesses(func_node, parent_class)

    # data_flows: return values, mutations, parameter usage
    data_flows = _extract_data_flows(func_node)

    meta: Dict = {
        'filepath': rel_path,
        'repo': repo,
        'lineno': func_node.lineno,
        'params': sig['params'],
        'returns': sig.get('returns'),
        'signature': signature_str,
        'source_code': source_code,
        'decorators': _get_decorators(func_node),
        'docstring': _get_docstring(func_node),
        'raises': exc['raises'],
        'catches': exc['catches'],
        'is_async': isinstance(func_node, ast.AsyncFunctionDef),
        'branches': _count_branches(func_node),
        'conditions': _extract_conditions(func_node),
        'data_flows': data_flows,
        # Populated for anything in a test file — covers test functions and
        # their helper classes (e.g. DomDocument.get_unique_child in pytest).
        # Production code is skipped to avoid noise from invariant asserts.
        'assert_patterns': _get_assert_patterns(func_node) if _is_test_file(rel_path) else [],
        'external_deps': external_deps,
        'side_effects': writes,
    }
    if parent_class is not None:
        meta['class'] = parent_class
    if parent_function is not None:
        meta['parent_function'] = parent_function
    return meta


def _is_type_checking_test(test: ast.AST) -> bool:
    """`TYPE_CHECKING`, `typing.TYPE_CHECKING` or any `x.TYPE_CHECKING`."""
    return ((isinstance(test, ast.Name) and test.id == 'TYPE_CHECKING')
            or (isinstance(test, ast.Attribute) and test.attr == 'TYPE_CHECKING'))


def _type_only_branch(test: ast.AST) -> Optional[str]:
    """Which branch of `if test:` only runs for a type checker: 'body' for
    `if TYPE_CHECKING:`, 'orelse' for `if not TYPE_CHECKING:`, else None."""
    if _is_type_checking_test(test):
        return 'body'
    if isinstance(test, ast.UnaryOp) and isinstance(test.op, ast.Not) and _is_type_checking_test(test.operand):
        return 'orelse'
    return None


class _ScopeImport(NamedTuple):
    """An import statement in a scope, with what decides which of its names
    the code can be using.

    type_only: in the body of `if TYPE_CHECKING:` (or the `else` of `if not
        TYPE_CHECKING:`): it never runs, so it must not replace a name the
        code really binds.
    shadowed: names the `try` body around this `except` handler imports.
        The handler's import of one of them is the fallback for when the
        body's failed; the body's is the one that normally runs.
    """
    stmt: ast.AST
    type_only: bool
    shadowed: FrozenSet[str]

    def bindings(self) -> List[Tuple[str, str]]:
        """(local name, qualified name) pairs, without the shadowed names."""
        return [(local, qualified) for local, qualified in _import_bindings(self.stmt)
                if local not in self.shadowed]


_NO_NAMES: FrozenSet[str] = frozenset()


def _iter_scope_imports_flagged(node: ast.AST, type_only: bool = False,
                                shadowed: FrozenSet[str] = _NO_NAMES):
    """Yield a _ScopeImport for each import in `node`'s own scope, in source
    order."""
    for child in ast.iter_child_nodes(node):
        yield from _flagged_scope_imports(child, type_only, shadowed)


def _flagged_scope_imports(child: ast.AST, type_only: bool, shadowed: FrozenSet[str]):
    if isinstance(child, (ast.Import, ast.ImportFrom)):
        yield _ScopeImport(child, type_only, shadowed)
    elif isinstance(child, _DEF_TYPES):
        return
    elif isinstance(child, ast.If) and _type_only_branch(child.test):
        which = _type_only_branch(child.test)
        for stmt in child.body:
            yield from _flagged_scope_imports(stmt, type_only or which == 'body', shadowed)
        for stmt in child.orelse:
            yield from _flagged_scope_imports(stmt, type_only or which == 'orelse', shadowed)
    elif isinstance(child, _TRY_TYPES):
        body_names = frozenset(
            local
            for stmt in child.body
            for found in _flagged_scope_imports(stmt, type_only, shadowed)
            if not found.type_only
            for local, _ in found.bindings()
        )
        for stmt in child.body:
            yield from _flagged_scope_imports(stmt, type_only, shadowed)
        for handler in child.handlers:
            yield from _iter_scope_imports_flagged(handler, type_only, shadowed | body_names)
        for stmt in child.orelse + child.finalbody:
            yield from _flagged_scope_imports(stmt, type_only, shadowed)
    elif isinstance(child, _SCOPE_TRANSPARENT):
        yield from _iter_scope_imports_flagged(child, type_only, shadowed)


def iter_scope_imports(node: ast.AST):
    """Yield the import statements that belong directly to `node`'s scope,
    in source order.

    Looks through compound statements (`try`/`except`, `if TYPE_CHECKING:`,
    `with`, loops, `match`), which don't introduce a scope, but not into a
    nested function or class: an import there is only visible inside it.
    """
    for found in _iter_scope_imports_flagged(node):
        yield found.stmt


def _import_bindings(stmt: Union[ast.Import, ast.ImportFrom]) -> List[Tuple[str, str]]:
    """(local name, fully qualified name) pairs an import statement binds."""
    if isinstance(stmt, ast.Import):
        # `import os.path` → local name is 'os', full name is 'os.path'
        return [(alias.asname or alias.name.split('.')[0], alias.name) for alias in stmt.names]
    mod = stmt.module or ''
    return [(alias.asname or alias.name, f"{mod}.{alias.name}" if mod else alias.name)
            for alias in stmt.names]


def _split_scope_imports(node: ast.AST) -> Tuple[Dict[str, str], Dict[str, str]]:
    """(runtime, type only) {local name: qualified name} for the imports in
    `node`'s own scope. A later import of a name replaces an earlier one,
    except that an `except` handler's import never replaces the `try` body's
    import of the same name."""
    runtime: Dict[str, str] = {}
    type_only: Dict[str, str] = {}
    for found in _iter_scope_imports_flagged(node):
        (type_only if found.type_only else runtime).update(found.bindings())
    return runtime, type_only


def _scope_import_map(node: ast.AST) -> Dict[str, str]:
    """{local_name: fully_qualified_name} for the imports in `node`'s own
    scope (see iter_scope_imports). A later import of the same local name
    replaces an earlier one, as it would when the code runs. An import under
    `if TYPE_CHECKING:` only supplies a name with no runtime import, so it
    never replaces the binding the code runs with."""
    runtime, type_only = _split_scope_imports(node)
    return {**type_only, **runtime}


def scope_import_map_over(outer: Dict[str, str], node: ast.AST) -> Dict[str, str]:
    """`outer` (the imports visible around `node`) plus the imports in `node`'s
    own scope. A runtime import of `node` replaces a name in `outer`; a type
    only one fills a name `outer` does not have, but never replaces one, so it
    can not reach a closure as the binding the code runs with."""
    runtime, type_only = _split_scope_imports(node)
    return {**type_only, **outer, **runtime}


def scope_import_bindings(node: ast.AST) -> Dict[str, List[Tuple[Tuple[int, int], str]]]:
    """{local name: [((line, col), qualified name), ...]} in source order for
    the runtime imports in `node`'s own scope. Type only imports are left
    out: they never run, so a call can not be going through them. So is an
    `except` handler's import of a name its `try` body imports."""
    bindings: Dict[str, List[Tuple[Tuple[int, int], str]]] = {}
    for found in _iter_scope_imports_flagged(node):
        if found.type_only:
            continue
        position = (found.stmt.lineno, found.stmt.col_offset)
        for local, qualified in found.bindings():
            bindings.setdefault(local, []).append((position, qualified))
    for hits in bindings.values():
        hits.sort(key=lambda hit: hit[0])
    return bindings


# A name bound to something other than an import (a parameter, an assignment,
# a def): what ScopeFlow records instead of a module path.
_LOCAL = None

_FlowState = Dict[str, Tuple[Optional[str], ...]]


def _merge_states(a: Optional[_FlowState], b: Optional[_FlowState]) -> Optional[_FlowState]:
    """The state after two paths join. None is a path that does not get there
    (it returned or raised). A name bound on only one path keeps that path's
    values: on the other it is unbound, which binds nothing."""
    if a is None:
        return b
    if b is None:
        return a
    merged = dict(a)
    for name, values in b.items():
        current = merged.get(name, ())
        merged[name] = current + tuple(v for v in values if v not in current)
    return merged


def _stored_names(node: ast.AST) -> List[str]:
    """Names a target expression (or a walrus inside an expression) binds in
    the enclosing scope. A comprehension's loop variables are its own scope in
    Python 3, so they are left out; a walrus inside a comprehension still
    binds in the enclosing scope, so it is kept."""
    in_comprehension: Set[int] = {
        id(n) for comp in ast.walk(node) if isinstance(comp, ast.comprehension)
        for n in ast.walk(comp.target)
    }
    return [n.id for n in ast.walk(node)
            if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)
            and id(n) not in in_comprehension]


class ScopeFlow:
    """What each name of one scope (a function, or a module) can be bound to
    at each point, following the paths through it: after an `if`/`elif`/
    `else` a name can hold what any branch (or, with no `else`, the code before
    it) bound; after a loop, what it held before or after an iteration; after
    `try`/`except`, the `try` body's import of a name rather than an `except`
    handler's fallback (the body's normally runs). A path that returns or
    raises does not reach the code after it. Values are module paths for
    imports and _LOCAL for any other binding. Nested functions and classes are
    other scopes: only the expressions evaluated where they are defined
    (decorators, defaults, bases) belong to this one.
    """

    def __init__(self, scope: ast.AST):
        self._at: Dict[int, _FlowState] = {}
        self._seen: Dict[str, Tuple[Optional[str], ...]] = {}
        self._outer: Set[str] = set()
        # >0 while walking a branch that never runs (`if TYPE_CHECKING:`):
        # its nodes are mapped, but what it binds is not recorded.
        self._never_runs = 0
        state: _FlowState = {}
        if isinstance(scope, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            args = scope.args
            for arg in args.posonlyargs + args.args + args.kwonlyargs + [args.vararg, args.kwarg]:
                if arg is not None:
                    state[arg.arg] = (_LOCAL,)
        body = scope.body if isinstance(scope.body, list) else [ast.Expr(scope.body)]
        self._record(state)
        self.end = self._block(body, state)

    def binds(self, name: str) -> bool:
        """Whether this scope binds `name` anywhere (so it is this scope's)."""
        return name in self._seen and name not in self._outer

    def at(self, node: ast.AST) -> Optional[_FlowState]:
        """The state just before the statement containing `node`, or None if
        `node` is not one this scope evaluates."""
        return self._at.get(id(node))

    def all_values(self, name: str) -> Tuple[Optional[str], ...]:
        """Everything `name` is bound to at any point (what a closure, which
        can run at any time, may see)."""
        return self._seen.get(name, ())

    def _record(self, state: _FlowState):
        if self._never_runs:
            return
        for name, values in state.items():
            current = self._seen.get(name, ())
            self._seen[name] = current + tuple(v for v in values if v not in current)

    def _map(self, node: Optional[ast.AST], state: _FlowState):
        if node is None:
            return
        for sub in ast.walk(node):
            self._at[id(sub)] = state

    def _bind(self, state: _FlowState, names, value) -> _FlowState:
        new = dict(state)
        for name in names:
            new[name] = (value,)
        self._record(new)
        return new

    def _block(self, stmts: List[ast.stmt], state: Optional[_FlowState]) -> Optional[_FlowState]:
        for stmt in stmts:
            if state is None:
                # Unreachable code: still map it, with nothing bound by flow.
                self._stmt(stmt, {})
                continue
            state = self._stmt(stmt, state)
        return state

    def _stmt(self, stmt: ast.stmt, state: _FlowState) -> Optional[_FlowState]:
        if isinstance(stmt, (ast.Import, ast.ImportFrom)):
            self._map(stmt, state)
            new = dict(state)
            for local, qualified in _import_bindings(stmt):
                new[local] = (qualified,)
            self._record(new)
            return new
        if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for expr in stmt.decorator_list + stmt.args.defaults + stmt.args.kw_defaults:
                self._map(expr, state)
            return self._bind(state, [stmt.name], _LOCAL)
        if isinstance(stmt, ast.ClassDef):
            for expr in stmt.decorator_list + stmt.bases + [k.value for k in stmt.keywords]:
                self._map(expr, state)
            return self._bind(state, [stmt.name], _LOCAL)
        if isinstance(stmt, (ast.Global, ast.Nonlocal)):
            self._outer.update(stmt.names)
            return state
        if isinstance(stmt, (ast.Return, ast.Raise)):
            self._map(stmt, state)
            return None
        if isinstance(stmt, ast.If):
            self._map(stmt.test, state)
            which = _type_only_branch(stmt.test)
            if which == 'body':
                self._never_runs += 1
                self._block(stmt.body, dict(state))
                self._never_runs -= 1
                return self._block(stmt.orelse, state) if stmt.orelse else state
            if which == 'orelse':
                self._never_runs += 1
                self._block(stmt.orelse, dict(state))
                self._never_runs -= 1
                return self._block(stmt.body, state)
            body = self._block(stmt.body, state)
            orelse = self._block(stmt.orelse, state) if stmt.orelse else state
            return _merge_states(body, orelse)
        if isinstance(stmt, (ast.For, ast.AsyncFor, ast.While)):
            self._map(stmt.iter if isinstance(stmt, (ast.For, ast.AsyncFor)) else stmt.test, state)
            entry = state
            if isinstance(stmt, (ast.For, ast.AsyncFor)):
                self._map(stmt.target, state)
                entry = self._bind(state, _stored_names(stmt.target), _LOCAL)
            after = _merge_states(state, self._block(stmt.body, entry))
            return self._block(stmt.orelse, after) if stmt.orelse else after
        if isinstance(stmt, (ast.With, ast.AsyncWith)):
            for item in stmt.items:
                self._map(item.context_expr, state)
            names = [n for item in stmt.items if item.optional_vars is not None
                     for n in _stored_names(item.optional_vars)]
            return self._block(stmt.body, self._bind(state, names, _LOCAL))
        if isinstance(stmt, _TRY_TYPES):
            body = self._block(stmt.body, state)
            body_names = {local for found in _iter_scope_imports_flagged(ast.Module(body=stmt.body, type_ignores=[]))
                          if not found.type_only for local, _ in found.bindings()}
            after = self._block(stmt.orelse, body) if stmt.orelse and body is not None else body
            for handler in stmt.handlers:
                self._map(handler.type, state)
                entry = self._bind(state, [handler.name], _LOCAL) if handler.name else state
                result = self._block(handler.body, entry)
                if result is not None and after is not None:
                    # The try body's import of a name is the one that normally
                    # runs; the handler's binding of it is the fallback.
                    result = {k: v for k, v in result.items() if k not in body_names}
                after = _merge_states(after, result)
            return self._block(stmt.finalbody, after) if stmt.finalbody else after
        if hasattr(ast, 'Match') and isinstance(stmt, ast.Match):
            self._map(stmt.subject, state)
            after = state
            for case in stmt.cases:
                captured = [n.name for n in ast.walk(case.pattern)
                            if isinstance(getattr(n, 'name', None), str)]
                captured += [n for p in ast.walk(case.pattern) for n in getattr(p, 'names', []) if isinstance(n, str)]
                entry = self._bind(state, captured, _LOCAL)
                self._map(case.guard, entry)
                after = _merge_states(after, self._block(case.body, entry))
            return after
        if isinstance(stmt, ast.Delete):
            self._map(stmt, state)
            new = dict(state)
            for target in stmt.targets:
                if isinstance(target, ast.Name):
                    new.pop(target.id, None)
            return new
        # Any other simple statement: an assignment, an expression. Its names
        # are evaluated before its own targets (and walrus targets) bind.
        self._map(stmt, state)
        stored = _stored_names(stmt)
        return self._bind(state, stored, _LOCAL) if stored else state


def scope_import_alternatives(node: ast.AST) -> Dict[str, List[str]]:
    """{local name: [module path, ...]} for the names in `node`'s own scope
    that can hold more than one import where the scope ends (an `if`/`else`
    importing a name in several branches, or an `if` that rebinds an earlier
    import): any of them can be what code after it, or a nested function,
    uses. Other names are left out (the import map has their one binding)."""
    flow = ScopeFlow(node)
    alternatives: Dict[str, List[str]] = {}
    names = flow.end if flow.end is not None else flow._seen
    for local, values in names.items():
        imports = [v for v in values if v is not _LOCAL]
        if len(imports) > 1:
            alternatives[local] = imports
    return alternatives


def scope_import_alternatives_over(outer: Dict[str, List[str]], node: ast.AST) -> Dict[str, List[str]]:
    """`outer` plus `node`'s own alternatives, as scope_import_map_over does
    for the import map: a name `node` imports at runtime replaces `outer`'s."""
    rebound = set(scope_import_bindings(node))
    merged = {name: names for name, names in outer.items() if name not in rebound}
    merged.update(scope_import_alternatives(node))
    return merged


class ImportScopes:
    """What a name means at a point in a function: the modules it can have
    been imported from, given the paths that reach that point (ScopeFlow).

    A function's own bindings decide for a name it binds: the ones that can
    reach the point of use, or, if none has yet, the binding outside it (so a
    call before `import b as client` still goes through the module's `import
    a as client`). A name bound in an enclosing function is looked up there,
    with any binding that function makes. `enclosing` is what the outermost
    function sees from outside (the module's imports, last one winning), and
    `enclosing_alternatives` lists, for a name of `enclosing` that can hold
    several imports, all of them (see scope_import_alternatives).
    """

    def __init__(self, enclosing: Dict[str, str],
                 enclosing_alternatives: Optional[Dict[str, List[str]]] = None):
        self.enclosing = enclosing
        self.enclosing_alternatives = enclosing_alternatives or {}
        self._flows: Dict[int, ScopeFlow] = {}

    def _flow(self, scope: ast.AST) -> ScopeFlow:
        if id(scope) not in self._flows:
            self._flows[id(scope)] = ScopeFlow(scope)
        return self._flows[id(scope)]

    def lookup(self, name: str, node: ast.AST, chain: List[ast.AST]) -> Optional[str]:
        """The qualified name `name` refers to at `node`, or None if it is not
        an imported name there. When it can be several modules (see
        lookup_all), the last one."""
        found = self.lookup_all(name, node, chain)
        return found[-1] if found else None

    def lookup_all(self, name: str, node: ast.AST, chain: List[ast.AST]) -> List[str]:
        """Every module `name` can be imported from at `node`, in the order
        its imports appear. Empty if it is not an imported name there, which
        includes a name the innermost function has bound locally (a parameter,
        an assignment) by that point. `chain` is the functions around `node`,
        outermost first (see call_scope_chains)."""
        for depth, scope in enumerate(reversed(chain)):
            flow = self._flow(scope)
            if not flow.binds(name):
                continue
            if depth == 0:
                state = flow.at(node)
                values = state.get(name) if state is not None else flow.all_values(name)
                if not values:
                    # Not bound on any path to here yet: the binding outside.
                    continue
            else:
                values = flow.all_values(name)
            return [v for v in values if v is not _LOCAL]
        if name in self.enclosing:
            return list(self.enclosing_alternatives.get(name) or [self.enclosing[name]])
        return []


def _collect_file_level_info(tree: ast.Module) -> Tuple[Dict[str, str], List[str], Dict[str, str]]:
    """Collect imports, __all__ exports, and module-level constants in one pass.

    Combining these into a single traversal avoids walking the module tree
    three separate times.

    Returns:
        import_map: {local_name: fully_qualified_name} for the imports in the
            module's own scope, including ones under try/except or
            `if TYPE_CHECKING:`. Imports inside a function are not here: they
            belong to that function (see _scope_import_map).
            Used to annotate call edges with the source module when the
            callee was imported (e.g. 'HTTPBasicAuth' → 'requests.auth.HTTPBasicAuth').
        exports: List of names in __all__, if defined.
        constants: {NAME: value_string} for UPPER_CASE module-level assignments.
    """
    import_map = _scope_import_map(tree)
    exports: List[str] = []
    constants: Dict[str, str] = {}

    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    if target.id == '__all__' and isinstance(node.value, (ast.List, ast.Tuple)):
                        exports = [
                            elt.value for elt in node.value.elts
                            if isinstance(elt, ast.Constant) and isinstance(elt.value, str)
                        ]
                    elif target.id.isupper():
                        unparsed = _safe_unparse(node.value)
                        if unparsed:
                            constants[target.id] = unparsed
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            # Annotated constant: MAX_SIZE: int = 100
            if node.target.id.isupper() and node.value:
                unparsed = _safe_unparse(node.value)
                if unparsed:
                    constants[node.target.id] = unparsed

    return import_map, exports, constants
