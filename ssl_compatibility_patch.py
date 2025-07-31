"""
Python 3.12 SSL Compatibility Patch for GoldGPT
Fixes the ssl.wrap_socket issue with eventlet and other libraries
"""

import ssl
import socket
from typing import Any, Optional

def patch_ssl_for_python312():
    """
    Apply SSL compatibility patch for Python 3.12+
    This fixes the missing ssl.wrap_socket function
    """
    if not hasattr(ssl, 'wrap_socket'):
        def wrap_socket(sock: socket.socket, 
                       keyfile: Optional[str] = None,
                       certfile: Optional[str] = None,
                       server_side: bool = False,
                       cert_reqs: int = ssl.CERT_NONE,
                       ssl_version: int = ssl.PROTOCOL_TLS,
                       ca_certs: Optional[str] = None,
                       do_handshake_on_connect: bool = True,
                       suppress_ragged_eofs: bool = True,
                       ciphers: Optional[str] = None) -> ssl.SSLSocket:
            """
            Replacement for deprecated ssl.wrap_socket function
            Uses modern ssl.SSLContext approach
            """
            context = ssl.SSLContext(ssl_version)
            
            if cert_reqs != ssl.CERT_NONE:
                context.check_hostname = False
            context.verify_mode = cert_reqs
            
            if certfile:
                context.load_cert_chain(certfile, keyfile)
            
            if ca_certs:
                context.load_verify_locations(ca_certs)
            
            if ciphers:
                context.set_ciphers(ciphers)
            
            return context.wrap_socket(
                sock,
                server_side=server_side,
                do_handshake_on_connect=do_handshake_on_connect,
                suppress_ragged_eofs=suppress_ragged_eofs
            )
        
        # Monkey patch the ssl module
        ssl.wrap_socket = wrap_socket
        print("✅ Applied Python 3.12 SSL compatibility patch")

def patch_eventlet_ssl():
    """
    Additional patches for eventlet SSL compatibility
    """
    try:
        import eventlet.green.ssl as green_ssl
        if not hasattr(green_ssl, 'wrap_socket') and hasattr(ssl, 'wrap_socket'):
            green_ssl.wrap_socket = ssl.wrap_socket
            print("✅ Applied eventlet SSL compatibility patch")
    except ImportError:
        pass  # eventlet not installed

# Apply patches immediately when module is imported
patch_ssl_for_python312()
patch_eventlet_ssl()
