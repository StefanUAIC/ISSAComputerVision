import datetime
import pickle
import socket
from typing import *

import select


class ObjectSocketParams:
    """Constants used by the ObjectSenderSocket and ObjectReceiverSocket classes"""
    OBJECT_HEADER_SIZE_BYTES = 4
    DEFAULT_TIMEOUT_S = 1
    CHUNK_SIZE_BYTES = 1024


class ObjectSenderSocket:
    """Socket that sends pickled objects to a receiver socket

    Attributes:
        ip (str): the IP address to bind the socket to
        port (int): the port to bind the socket to
        sock (socket.socket): the socket object used to listen for incoming connections
        conn (socket.socket): the connection socket object used after a receiver has connected
        print_when_awaiting_receiver (bool): whether to print messages when awaiting a receiver connection
        print_when_sending_object (bool): whether to print messages when sending an object
    """
    ip: str
    port: int
    sock: socket.socket
    conn: socket.socket
    print_when_awaiting_receiver: bool
    print_when_sending_object: bool

    def __init__(self, ip: str, port: int,
                 print_when_awaiting_receiver: bool = False,
                 print_when_sending_object: bool = False):
        """Initializes the ObjectSenderSocket and waits for a receiver to connect

        Args:
            ip (str): the IP address to bind the socket to
            port (int): the port to bind the socket to
            print_when_awaiting_receiver (bool): whether to print messages when awaiting a receiver connection
            print_when_sending_object (bool): whether to print messages when sending an object

        """
        self.ip = ip
        self.port = port

        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.bind((self.ip, self.port))
        self.conn = None

        self.print_when_awaiting_receiver = print_when_awaiting_receiver
        self.print_when_sending_object = print_when_sending_object

        self.await_receiver_conection()

    def await_receiver_conection(self):
        """Waits for a receiver to connect to the socket"""

        if self.print_when_awaiting_receiver:
            print(
                f'[{datetime.datetime.now()}][ObjectSenderSocket/{self.ip}:{self.port}] '
                f'awaiting receiver connection...')

        self.sock.listen(1)
        self.conn, _ = self.sock.accept()

        if self.print_when_awaiting_receiver:
            print(f'[{datetime.datetime.now()}][ObjectSenderSocket/{self.ip}:{self.port}] receiver connected')

    def close(self):
        """Closes the current connection"""
        self.conn.close()
        self.conn = None

    def is_connected(self) -> bool:
        """Checks if the socket is connected to a receiver

        Returns:
            bool: True if the socket is connected to a receiver, False otherwise
        """
        return self.conn is not None

    def send_object(self, obj: Any):
        """Sends a pickled object to the receiver

        Args:
            obj (Any): the object to send
        """
        data = pickle.dumps(obj)
        data_size = len(data)
        data_size_encoded = data_size.to_bytes(ObjectSocketParams.OBJECT_HEADER_SIZE_BYTES, 'little')
        self.conn.sendall(data_size_encoded)
        self.conn.sendall(data)
        if self.print_when_sending_object:
            print(
                f'[{datetime.datetime.now()}][ObjectSenderSocket/{self.ip}:{self.port}] '
                f'Sent object of size {data_size} bytes.')


class ObjectReceiverSocket:
    """A socket that receives pickled objects from a sender socket

    Attributes:
        ip (str): the IP address to connect to
        port (int): the port to connect to
        conn (socket.socket): the connection socket object used to communicate with the sender
        print_when_connecting_to_sender (bool): whether to print messages when connecting to the sender
        print_when_receiving_object (bool): whether to print messages when receiving an object
    """
    ip: str
    port: int
    conn: socket.socket
    print_when_connecting_to_sender: bool
    print_when_receiving_object: bool

    def __init__(self, ip: str, port: int,
                 print_when_connecting_to_sender: bool = False,
                 print_when_receiving_object: bool = False):
        """Initializes the ObjectReceiverSocket and connects to the sender

        Args:
            ip (str): the IP address to connect to
            port (int): the port to connect to
            print_when_connecting_to_sender (bool): whether to print messages when connecting to the sender
            print_when_receiving_object (bool): whether to print messages when receiving an object
        """
        self.ip = ip
        self.port = port
        self.print_when_connecting_to_sender = print_when_connecting_to_sender
        self.print_when_receiving_object = print_when_receiving_object

        self.connect_to_sender()

    def connect_to_sender(self):
        """Connects to the sender socket"""
        if self.print_when_connecting_to_sender:
            print(f'[{datetime.datetime.now()}][ObjectReceiverSocket/{self.ip}:{self.port}] connecting to sender...')

        self.conn = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.conn.connect((self.ip, self.port))

        if self.print_when_connecting_to_sender:
            print(f'[{datetime.datetime.now()}][ObjectReceiverSocket/{self.ip}:{self.port}] connected to sender')

    def close(self):
        """Closes the current connection"""
        self.conn.close()
        self.conn = None

    def is_connected(self) -> bool:
        """Checks if the socket is connected to a sender

        Returns:
            bool: True if the socket is connected to a sender, False otherwise
        """
        return self.conn is not None

    def recv_object(self) -> Any:
        """Receives a pickled object from the sender

        Returns:
            Any: the received object
        """
        obj_size_bytes = self._recv_object_size()
        data = self._recv_all(obj_size_bytes)
        obj = pickle.loads(data)
        if self.print_when_receiving_object:
            print(
                f'[{datetime.datetime.now()}][ObjectReceiverSocket/{self.ip}:{self.port}]'
                f' Received object of size {obj_size_bytes} bytes.')
        return obj

    def _recv_with_timeout(self, n_bytes: int,
                           timeout_s: float = ObjectSocketParams.DEFAULT_TIMEOUT_S) -> Optional[bytes]:
        """Receives a specified number of bytes from the socket with a timeout

        Args:
            n_bytes (int): the number of bytes to receive
            timeout_s (float): the timeout in seconds

        Returns:
            Optional[bytes]: the received bytes, or None if the timeout elapsed
        """
        rlist, _1, _2 = select.select([self.conn], [], [], timeout_s)
        if rlist:
            data = self.conn.recv(n_bytes)
            return data
        else:
            return None  # Only returned on timeout

    def _recv_all(self, n_bytes: int, timeout_s: float = ObjectSocketParams.DEFAULT_TIMEOUT_S) -> bytes:
        """Receives the number of bytes specified, waiting for each chunk with a timeout

        Args:
            n_bytes (int): the number of bytes to receive
            timeout_s (float): the timeout in seconds to wait for each chunk

        Returns:
            bytes: the received data as bytes

        Raises:
            socket.error: if the timeout elapsed without receiving all the data
        """
        data = []
        left_to_recv = n_bytes
        while left_to_recv > 0:
            desired_chunk_size = min(ObjectSocketParams.CHUNK_SIZE_BYTES, left_to_recv)
            chunk = self._recv_with_timeout(desired_chunk_size, timeout_s)
            if chunk is not None:
                data += [chunk]
                left_to_recv -= len(chunk)
            else:  # no more data incoming, timeout
                bytes_received = sum(map(len, data))
                raise socket.error(f'Timeout elapsed without any new data being received. '
                                   f'{bytes_received} / {n_bytes} bytes received.')
        data = b''.join(data)
        return data

    def _recv_object_size(self) -> int:
        """Receives the size of the upcoming object from the sender

        Returns:
            int: the size of the object in bytes
        """
        data = self._recv_all(ObjectSocketParams.OBJECT_HEADER_SIZE_BYTES)
        obj_size_bytes = int.from_bytes(data, 'little')
        return obj_size_bytes
