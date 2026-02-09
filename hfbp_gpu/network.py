#!/usr/bin/env python3
"""
THC Network Bridge (TCP Socket Server) — Temporal Holographic Computation
Accept external stimuli via socket
"""

import socket
import threading
from collections import deque
from pathlib import Path

from config import NETWORK_HOST, NETWORK_PORT, VERBOSE


class NetworkBridge:
    def __init__(self, engine):
        """Initialize network bridge."""
        self.engine = engine
        self.incoming = deque()  # Thread-safe for append/popleft
        self.running = False
        self._server_socket = None

        print(f"[NETWORK] Bridge init ({NETWORK_HOST}:{NETWORK_PORT})")

    def start(self):
        """Start server thread."""
        self.running = True
        self.server_thread = threading.Thread(target=self._server_loop, daemon=True)
        self.server_thread.start()
        print(f"[NETWORK] Server started on {NETWORK_HOST}:{NETWORK_PORT}")

    def _server_loop(self):
        """TCP server accept loop."""
        try:
            self._server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self._server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self._server_socket.settimeout(1.0)  # Allow periodic check of self.running
            self._server_socket.bind((NETWORK_HOST, NETWORK_PORT))
            self._server_socket.listen(5)

            while self.running:
                try:
                    c, addr = self._server_socket.accept()
                    try:
                        data = c.recv(4096).decode().strip()
                        if data:
                            self.incoming.append(data)
                            if VERBOSE:
                                print(f"[NETWORK] Received from {addr}: {data[:50]}")
                    finally:
                        c.close()
                except socket.timeout:
                    continue  # Check self.running again
                except OSError:
                    if self.running:
                        raise
                    break  # Socket closed during shutdown
        except Exception as e:
            print(f"[NETWORK] Server error: {e}")
        finally:
            if self._server_socket:
                try:
                    self._server_socket.close()
                except OSError:
                    pass

    def poll(self):
        """Non-blocking poll for messages. Called from engine loop."""
        while self.incoming:
            try:
                msg = self.incoming.popleft()  # O(1) with deque
            except IndexError:
                break  # Another thread consumed it
            self.engine.inject_text_stimulus(msg, gain=0.005)

    def stop(self):
        """Shutdown server."""
        self.running = False
        if self._server_socket:
            try:
                self._server_socket.close()
            except OSError:
                pass
        print("[NETWORK] Bridge stopped")


def test_network_client():
    """Test client: send stimulus via socket."""
    import sys

    host, port = NETWORK_HOST, NETWORK_PORT
    msg = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "test stimulus"

    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.connect((host, port))
        s.send(msg.encode())
        s.close()
        print(f"[CLIENT] Sent: {msg}")
    except Exception as e:
        print(f"[CLIENT] Error: {e}")


if __name__ == "__main__":
    test_network_client()
