"""
type_inference.py

Optional pyright-backed type resolution for factory-function call sites.

_get_factory_call_sites() (ast/helpers.py) records assignment sites where a
lowercase-named call's return value is invisible to the uppercase-heuristic
`uses`-edge detection (e.g. `session = requests.session()`). This module
answers "what type does that call actually return?" by driving a single
long-lived `pyright-langserver` process over stdio and asking it via the
standard `textDocument/hover` request — the same query an editor issues when
you hover a variable. No source files are rewritten; the extracted repo
tree is only ever read.

This is best-effort enrichment, not a required step: any failure (pyright
not installed, timeout, protocol error) results in an empty resolution map
rather than raising, so a KG build never fails because of this step.

Known limitation: hovering an *attribute* assignment (`obj.attr = call()`)
reports the attribute's already-declared type if one exists anywhere in the
class (e.g. `self.attr = None` in `__init__`), not the type of this
particular assignment's right-hand side. Resolution only succeeds when the
attribute has no prior type declared elsewhere in the class. This does not
affect bare local variable assignments (`x = call()`), which always resolve
based on the right-hand side.
"""

import json
import queue
import shutil
import subprocess
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# Wall-clock budget for starting pyright and getting through initialize.
STARTUP_TIMEOUT = 30
# Wall-clock budget per hover request.
HOVER_TIMEOUT = 10
# A hover that times out is skipped (pyright may still be analysing the
# project when the first ones arrive), but after this many in a row the
# server is treated as hung: the run stops and keeps what it has.
MAX_CONSECUTIVE_HOVER_TIMEOUTS = 3
# Wall-clock budget for the shutdown request at the end of a run.
SHUTDOWN_TIMEOUT = 5


class PyrightUnavailableError(Exception):
    """Raised when infer_types=True is requested but pyright is not installed."""


def is_available() -> bool:
    """Return True if the pyright-langserver executable can be found on PATH."""
    return shutil.which("pyright-langserver") is not None


class _RequestTimeout(Exception):
    """A request got no response within its deadline."""


class _LSPClient:
    """Minimal JSON-RPC-over-stdio client for a single pyright-langserver session.

    A background thread reads messages off the server's stdout into a queue,
    so request() can wait on the queue with a deadline. Reading the pipe
    directly would block forever on a server that hangs.
    """

    def __init__(self, proc: subprocess.Popen):
        self._proc = proc
        self._next_id = 1
        self._lock = threading.Lock()
        self._messages: "queue.Queue[Optional[Dict]]" = queue.Queue()
        threading.Thread(target=self._pump, daemon=True).start()

    def _pump(self) -> None:
        """Move messages from the server's stdout onto the queue until it
        closes. A None on the queue means the server is gone."""
        while True:
            try:
                message = self._read_message()
            except Exception:
                message = None
            self._messages.put(message)
            if message is None:
                return

    def _send(self, message: Dict) -> None:
        body = json.dumps(message).encode("utf-8")
        header = f"Content-Length: {len(body)}\r\n\r\n".encode("utf-8")
        self._proc.stdin.write(header + body)
        self._proc.stdin.flush()

    def _read_message(self) -> Optional[Dict]:
        stream = self._proc.stdout
        headers: Dict[str, str] = {}
        while True:
            line = stream.readline()
            if not line:
                return None
            if line in (b"\r\n", b"\n"):
                break
            decoded = line.decode("utf-8", errors="replace").strip()
            if ":" in decoded:
                key, _, value = decoded.partition(":")
                headers[key.strip()] = value.strip()
        length = int(headers.get("Content-Length", 0))
        if length == 0:
            return None
        body = stream.read(length)
        return json.loads(body.decode("utf-8", errors="replace"))

    def notify(self, method: str, params: Dict) -> None:
        self._send({"jsonrpc": "2.0", "method": method, "params": params})

    def request(self, method: str, params: Dict, timeout: float) -> Optional[Dict]:
        """Send a request and wait up to `timeout` seconds for its response.

        Diagnostics and other notifications may interleave before the
        response and are skipped, however many there are. A message that
        carries a `method` came from the server (a notification, or a
        request of its own that happens to reuse our id); it is never our
        response. Server requests are answered with an empty result so the
        server is not left waiting on us.

        Returns None if the server has exited. Raises _RequestTimeout if no
        response arrives in time (a late response is ignored by later
        requests, which match on their own id).
        """
        with self._lock:
            request_id = self._next_id
            self._next_id += 1
            self._send({"jsonrpc": "2.0", "id": request_id, "method": method, "params": params})
            deadline = time.monotonic() + timeout
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise _RequestTimeout(f"{method} timed out after {timeout}s")
                try:
                    message = self._messages.get(timeout=remaining)
                except queue.Empty:
                    raise _RequestTimeout(f"{method} timed out after {timeout}s") from None
                if message is None:
                    self._messages.put(None)  # keep later requests from waiting
                    return None
                if "method" in message:
                    if "id" in message:
                        self._send({"jsonrpc": "2.0", "id": message["id"], "result": None})
                    continue
                if message.get("id") == request_id:
                    return message


def resolve_types(
    repo_dir: Path,
    sites_by_file: Dict[str, List[Tuple[int, int]]],
) -> Dict[Tuple[str, int, int], str]:
    """Resolve inferred types at recorded factory-call assignment sites.

    Args:
        repo_dir: Root of the extracted source tree (read-only).
        sites_by_file: Dict mapping repo-relative file path to a list of
            (line, col) tuples (0-indexed) to query, as produced by
            _get_factory_call_sites().

    Returns:
        Dict mapping (rel_path, line, col) -> inferred type name (e.g.
        'Session'). Sites that pyright can't resolve to a simple type name
        (unions, unknown, builtins) are omitted rather than guessed at.
        Returns an empty dict if pyright is unavailable or fails outright —
        callers should treat that as "no enrichment," not an error.
    """
    if not sites_by_file or not is_available():
        return {}

    try:
        return _resolve_types_inner(repo_dir, sites_by_file)
    except Exception:
        # Any failure here (timeout, protocol hiccup, crash) degrades to
        # "no enrichment" rather than failing the whole KG build.
        return {}


def _resolve_types_inner(
    repo_dir: Path,
    sites_by_file: Dict[str, List[Tuple[int, int]]],
) -> Dict[Tuple[str, int, int], str]:
    resolved: Dict[Tuple[str, int, int], str] = {}
    clean_exit = False

    proc = subprocess.Popen(
        ["pyright-langserver", "--stdio"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        _drain_stderr_in_background(proc)
        client = _LSPClient(proc)

        # workspaceFolders + hover capabilities are required for pyright to
        # do full-project analysis (e.g. inferring an instance attribute's
        # type from an assignment in __init__). Without them pyright falls
        # back to shallow single-file analysis and reports such attributes
        # as "Unknown" even though its own CLI/reveal_type mode resolves
        # them correctly — confirmed by direct comparison during development.
        root_uri = repo_dir.resolve().as_uri()
        client.request("initialize", {
            "processId": None,
            "rootUri": root_uri,
            "workspaceFolders": [{"uri": root_uri, "name": repo_dir.name}],
            "capabilities": {
                "textDocument": {"hover": {"contentFormat": ["plaintext"]}},
                "workspace": {"workspaceFolders": True},
            },
        }, timeout=STARTUP_TIMEOUT)
        client.notify("initialized", {})
        consecutive_timeouts = 0

        for rel_path, sites in sites_by_file.items():
            abs_path = repo_dir / rel_path
            try:
                text = abs_path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue

            uri = abs_path.resolve().as_uri()
            client.notify("textDocument/didOpen", {
                "textDocument": {
                    "uri": uri, "languageId": "python", "version": 1, "text": text,
                }
            })

            for line, col in sites:
                try:
                    response = client.request("textDocument/hover", {
                        "textDocument": {"uri": uri},
                        "position": {"line": line, "character": col},
                    }, timeout=HOVER_TIMEOUT)
                except _RequestTimeout:
                    consecutive_timeouts += 1
                    if consecutive_timeouts >= MAX_CONSECUTIVE_HOVER_TIMEOUTS:
                        # The server looks hung. Keep what was resolved;
                        # the finally block kills the process.
                        return resolved
                    continue
                consecutive_timeouts = 0
                type_name = _parse_hover_type(response)
                if type_name:
                    resolved[(rel_path, line, col)] = type_name

            client.notify("textDocument/didClose", {"textDocument": {"uri": uri}})

        try:
            client.request("shutdown", {}, timeout=SHUTDOWN_TIMEOUT)
            client.notify("exit", {})
            clean_exit = True
        except (_RequestTimeout, OSError):
            pass
    finally:
        # Anything other than a clean shutdown (a timeout, an exception, an
        # early return) can leave a hung server behind; do not wait on it.
        if not clean_exit:
            proc.kill()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()

    return resolved


def _drain_stderr_in_background(proc: subprocess.Popen) -> None:
    """Consume stderr on a background thread so pyright never blocks on a full pipe."""
    def _drain():
        for _ in iter(proc.stderr.readline, b""):
            pass
    threading.Thread(target=_drain, daemon=True).start()


def _parse_hover_type(response: Optional[Dict]) -> Optional[str]:
    """Extract a simple class name from a pyright hover response, if any.

    Hover text looks like '(variable) s: Session'. Only single, simple
    (non-union, non-builtin) type names are returned — anything else
    (unions, 'Unknown', primitives) is treated as unresolved rather than
    guessed at, matching this module's "honest unknown" contract.
    """
    if not response or "result" not in response or response["result"] is None:
        return None

    contents = response["result"].get("contents")
    if isinstance(contents, dict):
        text = contents.get("value", "")
    elif isinstance(contents, str):
        text = contents
    else:
        return None

    if ":" not in text:
        return None
    type_part = text.rsplit(":", 1)[1].strip()
    # Strip markdown code fences pyright sometimes wraps hover text in.
    type_part = type_part.strip("`").strip()

    if not type_part or " " in type_part or "|" in type_part:
        return None  # union, unknown-with-qualifier, or otherwise not simple
    if type_part in ("Unknown", "None", "Any"):
        return None
    if type_part[0].islower():
        return None  # builtins (str, int, bool, ...) aren't repo classes

    return type_part
