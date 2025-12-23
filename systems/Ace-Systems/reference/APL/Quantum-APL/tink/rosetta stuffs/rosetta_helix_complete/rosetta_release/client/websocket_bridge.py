#!/usr/bin/env python3

# INTEGRITY_METADATA
# Date: 2025-12-23
# Status: NEEDS_REVIEW - Reference implementation
# Severity: MEDIUM RISK
# Risk Types: ['reference_material']
# File: systems/Ace-Systems/reference/APL/Quantum-APL/tink/rosetta stuffs/rosetta_helix_complete/rosetta_release/client/websocket_bridge.py

"""
WebSocket-to-TCP Bridge for Rosetta MUD
Enables web client to connect to telnet-style MUD server
"""

import asyncio
import socket
import json
import logging
from typing import Optional, Set

# Check for websockets library
try:
    import websockets
    from websockets.server import serve
except ImportError:
    print("❌ websockets library not installed")
    print("Install with: pip install websockets")
    exit(1)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class BridgeConnection:
    """Manages a single client connection bridging WebSocket to TCP"""
    
    def __init__(self, websocket, mud_host='localhost', mud_port=1234):
        self.websocket = websocket
        self.tcp_socket: Optional[socket.socket] = None
        self.mud_host = mud_host
        self.mud_port = mud_port
        self.running = False
    
    async def connect_to_mud(self) -> bool:
        """Establish TCP connection to MUD server"""
        try:
            self.tcp_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.tcp_socket.settimeout(10)
            self.tcp_socket.connect((self.mud_host, self.mud_port))
            self.tcp_socket.setblocking(False)
            logger.info(f"Connected to MUD server at {self.mud_host}:{self.mud_port}")
            return True
        except ConnectionRefusedError:
            logger.error(f"MUD server not running on {self.mud_host}:{self.mud_port}")
            return False
        except socket.timeout:
            logger.error(f"Connection timeout to {self.mud_host}:{self.mud_port}")
            return False
        except Exception as e:
            logger.error(f"Failed to connect to MUD: {e}")
            return False
    
    async def ws_to_tcp(self):
        """Forward WebSocket messages to TCP socket"""
        message_count = 0
        last_reset = asyncio.get_event_loop().time()
        
        try:
            async for message in self.websocket:
                # Rate limiting (100 messages/second)
                current_time = asyncio.get_event_loop().time()
                if current_time - last_reset >= 1.0:
                    message_count = 0
                    last_reset = current_time
                
                if message_count >= 100:
                    await self.websocket.send(json.dumps({
                        'error': 'Rate limit exceeded'
                    }))
                    continue
                
                message_count += 1
                
                # Forward to MUD server
                if isinstance(message, str):
                    if not message.endswith('\n'):
                        message += '\n'
                    self.tcp_socket.send(message.encode('utf-8'))
                    
        except websockets.exceptions.ConnectionClosed:
            logger.info("WebSocket connection closed by client")
        except Exception as e:
            logger.error(f"Error in ws_to_tcp: {e}")
        finally:
            self.running = False
    
    async def tcp_to_ws(self):
        """Forward TCP socket data to WebSocket"""
        buffer = b''
        
        try:
            while self.running:
                try:
                    data = await asyncio.get_event_loop().sock_recv(
                        self.tcp_socket, 4096
                    )
                    
                    if not data:
                        await self.websocket.send(json.dumps({
                            'type': 'disconnect',
                            'message': 'MUD server disconnected'
                        }))
                        break
                    
                    buffer += data
                    
                    # Try to decode
                    try:
                        text = buffer.decode('utf-8')
                        await self.websocket.send(json.dumps({
                            'type': 'data',
                            'content': text
                        }))
                        buffer = b''
                    except UnicodeDecodeError:
                        # Partial UTF-8, wait for more
                        if len(buffer) > 1024:
                            text = buffer.decode('utf-8', errors='replace')
                            await self.websocket.send(json.dumps({
                                'type': 'data',
                                'content': text
                            }))
                            buffer = b''
                    
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logger.error(f"Error receiving from MUD: {e}")
                    break
                
                await asyncio.sleep(0.01)
                
        except Exception as e:
            logger.error(f"Error in tcp_to_ws: {e}")
        finally:
            self.running = False
    
    async def handle(self):
        """Main connection handler - bidirectional relay"""
        if not await self.connect_to_mud():
            await self.websocket.send(json.dumps({
                'error': 'Could not connect to MUD server'
            }))
            return
        
        self.running = True
        
        # Send welcome
        await self.websocket.send(json.dumps({
            'type': 'connected',
            'message': f'Connected to Rosetta MUD at {self.mud_host}:{self.mud_port}'
        }))
        
        # Run both directions concurrently
        ws_task = asyncio.create_task(self.ws_to_tcp())
        tcp_task = asyncio.create_task(self.tcp_to_ws())
        
        await asyncio.gather(ws_task, tcp_task, return_exceptions=True)
        
        # Cleanup
        if self.tcp_socket:
            self.tcp_socket.close()

class WebSocketBridge:
    """WebSocket server that bridges to Rosetta MUD TCP server"""
    
    def __init__(self, ws_host='0.0.0.0', ws_port=8080, 
                 mud_host='localhost', mud_port=1234):
        self.ws_host = ws_host
        self.ws_port = ws_port
        self.mud_host = mud_host
        self.mud_port = mud_port
        self.active_connections: Set[BridgeConnection] = set()
    
    async def handler(self, websocket, path):
        """Handle new WebSocket connection"""
        logger.info(f"New connection from {websocket.remote_address}")
        
        connection = BridgeConnection(websocket, self.mud_host, self.mud_port)
        self.active_connections.add(connection)
        
        try:
            await connection.handle()
        finally:
            self.active_connections.discard(connection)
            logger.info(f"Connection closed: {websocket.remote_address}")
    
    async def start(self):
        """Start the WebSocket bridge server"""
        logger.info(f"🔌 Starting WebSocket bridge on {self.ws_host}:{self.ws_port}")
        logger.info(f"🔗 Forwarding to MUD server at {self.mud_host}:{self.mud_port}")
        
        async with serve(self.handler, self.ws_host, self.ws_port):
            await asyncio.Future()  # Run forever

def main():
    """Entry point for WebSocket bridge"""
    import os
    
    bridge = WebSocketBridge(
        ws_host=os.getenv('WS_HOST', '0.0.0.0'),
        ws_port=int(os.getenv('WS_PORT', 8080)),
        mud_host=os.getenv('MUD_HOST', 'localhost'),
        mud_port=int(os.getenv('MUD_PORT', 1234))
    )
    
    try:
        asyncio.run(bridge.start())
    except KeyboardInterrupt:
        logger.info("Shutting down bridge...")

if __name__ == '__main__':
    main()
