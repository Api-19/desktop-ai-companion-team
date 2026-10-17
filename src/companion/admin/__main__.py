from pathlib import Path
from wsgiref.simple_server import WSGIRequestHandler, WSGIServer, make_server

from companion.admin.app import AdminConfigError, build_admin_app, load_admin_config
from companion.runtime.settings import SettingsError, load_settings


class _QuietHandler(WSGIRequestHandler):
    def log_message(self, format, *args):
        return


class _QuietServer(WSGIServer):
    request_timeout = 2.0

    def get_request(self):
        connection, address = super().get_request()
        connection.settimeout(self.request_timeout)
        return connection, address

    def handle_error(self, request, client_address):
        return


def main() -> int:
    try:
        settings = load_settings(Path.cwd())
        config = load_admin_config(settings)
    except (SettingsError, AdminConfigError):
        print("Admin unavailable")
        return 2
    with make_server(
        "127.0.0.1",
        0,
        lambda _e, _s: [],
        server_class=_QuietServer,
        handler_class=_QuietHandler,
    ) as server:
        try:
            app = build_admin_app(settings, config, port=server.server_port)
        except Exception:
            print("Admin unavailable")
            return 2
        server.set_app(app)
        code = app.issue_start_code()
        print(f"Open http://127.0.0.1:{server.server_port}/admin")
        print(f"One-time start code: {code}")
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            app.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
