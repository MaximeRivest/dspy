"""A parent-owned egress broker for isolated scoring children (Q7).

The broker is the honest answer to "let the child reach a few endpoints"
(the notes reject IP allowlists — anycast and CDNs make them rot). It is
a localhost forward/CONNECT proxy the PARENT owns, with three properties
the notes call the killer features:

- **Hostname allowlist.** Only the declared egress hosts are reachable;
  everything else is refused AND logged. Deny-by-default.
- **Per-request log.** Every request (host, method, allowed/denied) is
  recorded, so "what did the child reach" is one list to read, never a
  packet-capture exercise.
- **Credential injection.** The child gets `HTTPS_PROXY`/`HTTP_PROXY` and
  NO credential env vars; the broker attaches the `Authorization` header
  on egress to an allowlisted host. Generated code cannot leak a key it
  never held.

This is optimizer machinery in the PARENT, not generated code — it holds
the keys and never runs in the child. It lives under `dspy/optim`
(alongside the scoring machinery it serves) rather than in the engine,
because it is a FlexIR-loop concern: the engine materializes and runs
artifacts, while the broker is a scoring-time supervisor the optimizer
owns. It handles HTTPS via `CONNECT` (tunnelling to allowlisted hosts,
where TLS is end-to-end so no header can be injected) and plain HTTP by
forwarding (where the credential header CAN be injected). For the
localhost-stub tests the child speaks plain HTTP through the forward
path, so injection is observable.
"""

from __future__ import annotations

import http.client
import socket
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit

__all__ = ["EgressBroker"]


class EgressBroker:
    """A localhost forward proxy: hostname allowlist + log + credential inject.

    Args:
        allow_hosts: Hostnames the child may reach. Anything else is
            refused (HTTP 403) and logged.
        inject: `{hostname: {"header": name, "value": secret}}` — the
            broker attaches `header: value` on forwarded requests to that
            host, so the secret never enters the child. Applied only on
            the plain-HTTP forward path (HTTPS CONNECT is an opaque
            tunnel; inject over plain HTTP where the header is ours to
            add).

    Attributes:
        requests: One record per request:
            `{"host", "method", "allowed", "injected"}`, in order.
    """

    def __init__(self, allow_hosts: frozenset[str], inject: dict[str, dict[str, str]] | None = None):
        self.allow_hosts = frozenset(allow_hosts)
        self.inject = inject or {}
        self.requests: list[dict[str, object]] = []
        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None

    @property
    def proxy_url(self) -> str:
        assert self._server is not None, "broker is not started"
        host, port = self._server.server_address[:2]
        return f"http://{host}:{port}"

    def start(self) -> EgressBroker:
        broker = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *args):
                pass

            def do_CONNECT(self):  # HTTPS tunnels: allowlist only, no inject
                host = self.path.rsplit(":", 1)[0]
                allowed = broker._host_allowed(host)
                broker.requests.append({"host": host, "method": "CONNECT", "allowed": allowed, "injected": False})
                if not allowed:
                    self.send_error(403, "host not on the broker allowlist")
                    return
                broker._tunnel(self)

            def do_GET(self):
                broker._forward(self, "GET")

            def do_POST(self):
                broker._forward(self, "POST")

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
            self._server = None
        if self._thread is not None:
            self._thread.join(timeout=5)
            self._thread = None

    def _host_allowed(self, host: str) -> bool:
        return host in self.allow_hosts

    def _forward(self, handler: BaseHTTPRequestHandler, method: str) -> None:
        parts = urlsplit(handler.path)
        host = parts.hostname or ""
        allowed = self._host_allowed(host)
        injected = False
        if not allowed:
            self.requests.append({"host": host, "method": method, "allowed": False, "injected": False})
            handler.send_error(403, "host not on the broker allowlist")
            return
        length = int(handler.headers.get("Content-Length", 0))
        body = handler.rfile.read(length) if length else None
        headers = {key: value for key, value in handler.headers.items() if key.lower() != "proxy-connection"}
        grant = self.inject.get(host)
        if grant is not None:
            headers[grant["header"]] = grant["value"]
            injected = True
        self.requests.append({"host": host, "method": method, "allowed": True, "injected": injected})
        port = parts.port or 80
        upstream = http.client.HTTPConnection(host, port, timeout=30)
        try:
            path = parts.path or "/"
            if parts.query:
                path += "?" + parts.query
            upstream.request(method, path, body=body, headers=headers)
            response = upstream.getresponse()
            payload = response.read()
            handler.send_response(response.status)
            for key, value in response.getheaders():
                if key.lower() in ("transfer-encoding", "connection"):
                    continue
                handler.send_header(key, value)
            handler.send_header("Content-Length", str(len(payload)))
            handler.end_headers()
            handler.wfile.write(payload)
        finally:
            upstream.close()

    def _tunnel(self, handler: BaseHTTPRequestHandler) -> None:  # pragma: no cover - HTTPS path
        host, port = handler.path.rsplit(":", 1)
        upstream = socket.create_connection((host, int(port)), timeout=30)
        handler.send_response(200, "Connection established")
        handler.end_headers()
        client = handler.connection
        sockets = [client, upstream]
        import select

        while True:
            readable, _, _ = select.select(sockets, [], [], 30)
            if not readable:
                break
            for source in readable:
                other = upstream if source is client else client
                data = source.recv(65536)
                if not data:
                    upstream.close()
                    return
                other.sendall(data)
