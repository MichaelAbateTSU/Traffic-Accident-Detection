"""HTTPS CONNECT proxy: exact host allowlist, public IPs and pinned connections."""

import ipaddress
import logging
import os
import select
import socket
import socketserver
import threading
import time
from urllib.parse import urlsplit

logger = logging.getLogger(__name__)

def tunnel_lifetime_seconds(job_timeout_seconds: int) -> int:
    if not 1 <= job_timeout_seconds <= 3600:
        raise ValueError("JOB_TIMEOUT_SECONDS must be within 1-3600")
    return job_timeout_seconds + 30


def resolve_public(host: str, port: int, allowed_hosts: set[str]) -> list[tuple]:
    if host.lower() not in allowed_hosts or port != 443:
        raise ValueError("Host or port is not approved")
    addresses = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    if not addresses or any(
        not ipaddress.ip_address(address[4][0]).is_global
        or ipaddress.ip_address(address[4][0]).is_multicast
        or address[4][0] == "168.63.129.16"
        for address in addresses
    ):
        raise ValueError("Destination is not exclusively public")
    return addresses


def connect_public(host: str, port: int, allowed_hosts: set[str]) -> socket.socket:
    # Use the validated numeric sockaddr; never resolve the name again at connect.
    addresses = resolve_public(host, port, allowed_hosts)
    last_error = None
    for family, socktype, proto, _, sockaddr in addresses:
        upstream = socket.socket(family, socktype, proto)
        upstream.settimeout(10)
        try:
            upstream.connect(sockaddr)
            return upstream
        except OSError as exc:
            upstream.close()
            last_error = exc
    raise OSError("No approved upstream address could be reached") from last_error


class ProxyHandler(socketserver.StreamRequestHandler):
    def handle(self) -> None:
        self.connection.settimeout(10)
        try:
            line = self.rfile.readline(2049)
            if len(line) > 2048:
                raise ValueError("Request line too long")
            method, target, version = line.decode("ascii").strip().split()
            if method != "CONNECT" or version not in {"HTTP/1.0", "HTTP/1.1"}:
                raise ValueError("Only HTTPS CONNECT is supported")
            parsed = urlsplit(f"//{target}")
            if not parsed.hostname or parsed.username or parsed.path or parsed.query or parsed.fragment:
                raise ValueError("Invalid CONNECT target")
            total = 0
            while True:
                header = self.rfile.readline(8193)
                total += len(header)
                if not header or total > 8192:
                    raise ValueError("Invalid proxy headers")
                if header in (b"\r\n", b"\n"):
                    break
            upstream = connect_public(parsed.hostname, parsed.port or 443, self.server.allowed_hosts)
        except (ValueError, UnicodeError, OSError):
            logger.exception("Rejected or failed proxy connection")
            self.connection.sendall(b"HTTP/1.1 403 Forbidden\r\nContent-Length: 0\r\n\r\n")
            return
        with upstream:
            self.connection.sendall(b"HTTP/1.1 200 Connection Established\r\n\r\n")
            deadline = time.monotonic() + self.server.tunnel_lifetime
            try:
                while time.monotonic() < deadline:
                    readable, _, _ = select.select(
                        [self.connection, upstream], [], [], max(0, min(10, deadline - time.monotonic())),
                    )
                    if not readable:
                        logger.warning("Closing idle proxy tunnel")
                        return
                    for source in readable:
                        data = source.recv(65536)
                        if not data:
                            return
                        destination = upstream if source is self.connection else self.connection
                        destination.sendall(data)
                logger.warning("Closing proxy tunnel at its lifetime limit")
            except OSError:
                logger.exception("Proxy tunnel transport failed")


class ProxyServer(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True
    request_queue_size = 16

    def __init__(
        self, address: tuple, allowed_hosts: set[str], *, job_timeout_seconds: int = 300,
    ) -> None:
        self.allowed_hosts = allowed_hosts
        self.tunnel_lifetime = tunnel_lifetime_seconds(job_timeout_seconds)
        self.slots = threading.BoundedSemaphore(16)
        super().__init__(address, ProxyHandler)

    def verify_request(self, request, client_address) -> bool:
        acquired = self.slots.acquire(blocking=False)
        if not acquired:
            logger.warning("Proxy connection capacity exhausted")
        return acquired

    def process_request_thread(self, request, client_address) -> None:
        try:
            super().process_request_thread(request, client_address)
        finally:
            self.slots.release()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    hosts = {host.strip().lower() for host in os.environ.get("STREAM_HOSTS", "").split(",") if host.strip()}
    if not hosts:
        raise SystemExit("STREAM_HOSTS must explicitly list approved camera/CDN hosts")
    with ProxyServer(
        ("0.0.0.0", 3128), hosts,
        job_timeout_seconds=int(os.environ.get("JOB_TIMEOUT_SECONDS", "300")),
    ) as server:
        server.serve_forever()
