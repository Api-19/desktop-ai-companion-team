import socket
import threading
from wsgiref.simple_server import make_server

from companion.admin.__main__ import _QuietHandler, _QuietServer


def test_idle_preconnection_does_not_block_following_request(monkeypatch):
    monkeypatch.setattr(_QuietServer, "request_timeout", 0.1, raising=False)
    accepted = threading.Event()

    class ObservedServer(_QuietServer):
        def get_request(self):
            result = super().get_request()
            accepted.set()
            return result

    def app(environ, start_response):
        start_response("200 OK", [("Content-Type", "text/plain")])
        return [b"synthetic"]

    server = make_server(
        "127.0.0.1", 0, app, server_class=ObservedServer, handler_class=_QuietHandler
    )
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    idle = socket.create_connection(server.server_address, timeout=1)
    try:
        assert accepted.wait(1)
        with socket.create_connection(server.server_address, timeout=1) as normal:
            normal.sendall(b"GET / HTTP/1.0\r\nHost: localhost\r\n\r\n")
            assert normal.recv(4096).startswith(b"HTTP/1.0 200 OK")
    finally:
        idle.close()
        server.shutdown()
        server.server_close()
        worker.join(timeout=1)
