"""A local HTTP file server that misbehaves on request, for the download and pipeline tests.

It serves one payload with Range support, an ETag and Last-Modified, like the
real host. Switches make it cut the connection mid-body, ignore Range, change
its ETag, or answer a Range request at the wrong offset. Every request is
recorded, so a test can assert exactly what went over the wire.
"""

from __future__ import annotations

import re
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Iterator

import pytest

_RANGE = re.compile(r"bytes=(\d+)-")


class FileServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self) -> None:
        super().__init__(("127.0.0.1", 0), _Handler)
        self.payload = b""
        self.etag = '"v1"'
        self.send_etag = True
        #: Absolute byte offset at which the next GET is cut off (once).
        self.drop_after: int | None = None
        self.ignore_range = False
        #: Added to the start of every Content-Range, to fake a misbehaving server.
        self.range_skew = 0
        #: ETag to switch to once the next GET has been answered.
        self.etag_after_get: str | None = None
        self.requests: list[tuple[str, str | None]] = []

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.server_address[1]}/Dataset.zip"

    def gets(self) -> list[str | None]:
        return [range_header for method, range_header in self.requests if method == "GET"]

    def handle_error(self, request: object, client_address: tuple[str, int]) -> None:
        # A client that rejects a response after its headers (changed ETag, bad
        # Content-Range) closes the socket mid-body. That is the behaviour under
        # test, not a server fault; anything else is still reported.
        if isinstance(sys.exc_info()[1], ConnectionError):
            return
        super().handle_error(request, client_address)


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server: FileServer

    def log_message(self, format: str, *args: object) -> None:  # noqa: A002 - stdlib signature
        pass

    def _send_headers(self, status: int, length: int, start: int = 0) -> None:
        srv = self.server
        self.send_response(status)
        self.send_header("Content-Length", str(length))
        if srv.send_etag:
            self.send_header("ETag", srv.etag)
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Last-Modified", "Thu, 11 Aug 2022 15:01:09 GMT")
        if status == 206:
            total = len(srv.payload)
            self.send_header("Content-Range", f"bytes {start + srv.range_skew}-{total - 1}/{total}")
        self.end_headers()

    def do_HEAD(self) -> None:  # noqa: N802 - stdlib naming
        self.server.requests.append(("HEAD", None))
        self._send_headers(200, len(self.server.payload))

    def do_GET(self) -> None:  # noqa: N802 - stdlib naming
        srv = self.server
        range_header = self.headers.get("Range")
        srv.requests.append(("GET", range_header))

        start = 0
        if range_header and not srv.ignore_range:
            if_range = self.headers.get("If-Range")
            match = _RANGE.fullmatch(range_header)
            if match and (if_range is None or if_range == srv.etag):
                start = int(match.group(1))

        body = srv.payload[start:]
        self._send_headers(206 if start else 200, len(body), start)

        cut = srv.drop_after
        if cut is not None and cut > start:
            srv.drop_after = None
            self.wfile.write(body[: cut - start])
            self.wfile.flush()
            self.close_connection = True
        else:
            self.wfile.write(body)

        if srv.etag_after_get is not None:
            srv.etag, srv.etag_after_get = srv.etag_after_get, None


@pytest.fixture
def file_server() -> Iterator[FileServer]:
    server = FileServer()
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server
    finally:
        server.shutdown()
        server.server_close()
