"""Tests for the pyright client's deadlines and message handling.

kg/type_inference.py drives a long lived pyright-langserver process over
stdio. Two problems made a KG build with infer_types=True fragile:

* STARTUP_TIMEOUT and HOVER_TIMEOUT were defined but never enforced, so a
  server that hung left the build blocked forever on a read.
* request() read at most 50 messages looking for its response (a burst of
  diagnostics notifications defeated it), and mistook a server initiated
  request that happened to share the response's id for the response itself.

The tests run the real client against a small fake language server (a
Python script) that can hang, die, flood notifications, or send its own
requests, so none of this depends on pyright being installed.
"""

import sys
import time
from pathlib import Path

import pytest

from kg_construction.kg import type_inference

FAKE_SERVER = r'''
import json, os, sys, time

mode = os.environ.get("FAKE_LSP_MODE", "ok")
log = os.environ.get("FAKE_LSP_LOG")


def note(text):
    if log:
        with open(log, "a") as f:
            f.write(text + "\n")


def read():
    headers = {}
    while True:
        line = sys.stdin.buffer.readline()
        if not line:
            return None
        if line in (b"\r\n", b"\n"):
            break
        key, _, value = line.decode().partition(":")
        headers[key.strip()] = value.strip()
    return json.loads(sys.stdin.buffer.read(int(headers.get("Content-Length", 0))))


def send(message):
    body = json.dumps(message).encode()
    sys.stdout.buffer.write(b"Content-Length: %d\r\n\r\n" % len(body) + body)
    sys.stdout.buffer.flush()


hovers = 0
while True:
    m = read()
    if m is None:
        break
    method, mid = m.get("method"), m.get("id")
    if method is None and mid is not None:
        note("reply:%s" % mid)  # the client answered one of our requests
        continue
    if method == "initialize":
        if mode == "hang_init":
            time.sleep(60)
        send({"jsonrpc": "2.0", "id": mid, "result": {"capabilities": {}}})
    elif method == "textDocument/hover":
        hovers += 1
        note("hover:%s" % mid)
        if mode == "hang_hover" or (mode == "hang_after_first" and hovers > 1):
            continue  # keep reading requests, but never answer this one
        if mode == "die_on_hover":
            sys.exit(0)
        if mode == "noisy":
            for _ in range(80):
                send({"jsonrpc": "2.0", "method": "textDocument/publishDiagnostics", "params": {}})
        if mode == "server_request":
            send({"jsonrpc": "2.0", "id": mid, "method": "workspace/configuration",
                  "params": {"items": []}})
        send({"jsonrpc": "2.0", "id": mid,
              "result": {"contents": {"kind": "plaintext", "value": "(variable) s: Session"}}})
    elif method == "shutdown":
        send({"jsonrpc": "2.0", "id": mid, "result": None})
    elif method == "exit":
        break
'''


@pytest.fixture
def fake_pyright(tmp_path, monkeypatch):
    """Point the client at the fake server and shrink the deadlines."""
    server = tmp_path / "fake_lsp.py"
    server.write_text(FAKE_SERVER)
    log = tmp_path / "lsp.log"

    real_popen = type_inference.subprocess.Popen

    def popen(args, **kwargs):
        return real_popen([sys.executable, str(server)], **kwargs)

    monkeypatch.setattr(type_inference.subprocess, "Popen", popen)
    monkeypatch.setattr(type_inference, "is_available", lambda: True)
    monkeypatch.setattr(type_inference, "STARTUP_TIMEOUT", 1.5)
    monkeypatch.setattr(type_inference, "HOVER_TIMEOUT", 0.5)
    monkeypatch.setenv("FAKE_LSP_LOG", str(log))

    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "m.py").write_text("a = 1\nb = 2\nc = 3\nd = 4\ne = 5\nf = 6\n")

    class Handle:
        repo_dir = repo

        def run(self, mode, n_sites=1):
            monkeypatch.setenv("FAKE_LSP_MODE", mode)
            sites = {"m.py": [(i, 0) for i in range(n_sites)]}
            start = time.monotonic()
            result = type_inference.resolve_types(repo, sites)
            return result, time.monotonic() - start

        def log_lines(self):
            return log.read_text().splitlines() if log.exists() else []

    return Handle()


class TestTimeoutsAreEnforced:
    def test_normal_server_resolves_every_site(self, fake_pyright):
        result, _ = fake_pyright.run("ok", n_sites=3)
        assert result == {("m.py", i, 0): "Session" for i in range(3)}

    def test_server_that_hangs_on_startup_gives_empty_result_in_time(self, fake_pyright):
        result, elapsed = fake_pyright.run("hang_init")
        assert result == {}
        assert elapsed < 10

    def test_server_that_hangs_on_hover_is_abandoned_after_repeated_timeouts(self, fake_pyright):
        result, elapsed = fake_pyright.run("hang_hover", n_sites=6)
        assert result == {}
        assert elapsed < 10
        hovers = [line for line in fake_pyright.log_lines() if line.startswith("hover:")]
        # Gave up after the allowed number of consecutive timeouts, rather
        # than waiting on all six.
        assert len(hovers) == type_inference.MAX_CONSECUTIVE_HOVER_TIMEOUTS

    def test_results_gathered_before_a_hang_are_kept(self, fake_pyright):
        result, elapsed = fake_pyright.run("hang_after_first", n_sites=5)
        assert result == {("m.py", 0, 0): "Session"}
        assert elapsed < 10

    def test_server_that_exits_mid_run_does_not_hang_the_build(self, fake_pyright):
        result, elapsed = fake_pyright.run("die_on_hover", n_sites=3)
        assert result == {}
        assert elapsed < 10

    def test_burst_of_notifications_does_not_lose_the_response(self, fake_pyright):
        # 80 notifications precede the response; the old client stopped
        # looking after 50 messages and silently dropped the site.
        result, _ = fake_pyright.run("noisy")
        assert result == {("m.py", 0, 0): "Session"}


class TestServerInitiatedRequests:
    def test_server_request_sharing_the_responses_id_is_not_mistaken_for_it(self, fake_pyright):
        result, _ = fake_pyright.run("server_request")
        assert result == {("m.py", 0, 0): "Session"}

    def test_client_answers_server_requests_so_the_server_is_not_left_waiting(self, fake_pyright):
        fake_pyright.run("server_request")
        hover_id = next(
            line.split(":")[1] for line in fake_pyright.log_lines() if line.startswith("hover:"))
        assert f"reply:{hover_id}" in fake_pyright.log_lines()
