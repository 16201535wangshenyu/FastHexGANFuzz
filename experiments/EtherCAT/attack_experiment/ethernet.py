import struct
import socket
import uuid


class EthernetSocket(object):
    # Constants from sys/ethernet.h
    ETH_P_IP = 0x0800
    ETH_P_ARP = 0x0806
    ETH_P_ALL = 0x0003
    ETH_CAT = 0x88a4

    def __init__(self):

        self.broadcast_mac = b"\xff\xff\xff\xff\xff\xff"

        # We can only send to the broadcast mac until we know where the gateway is
        self.dest_mac = self.broadcast_mac
        self.src_mac = b"\x00\x33\x31\x2d\xdd\x6c"

        # For my arch machine, use eth0 for everything else
        #self.interface = "ens33"
        self.interface = "ens33"

        # Construct sockets. Due to linux oddities, there needs to be a separate socket for receiving and sending
        self.send_sock = socket.socket(
            socket.AF_PACKET, socket.SOCK_RAW, socket.htons(EthernetSocket.ETH_CAT))
        self.recv_sock = socket.socket(
            socket.AF_PACKET, socket.SOCK_RAW, socket.htons(EthernetSocket.ETH_CAT))
        self.recv_sock.setblocking(0)
        self.recv_sock.settimeout(1)  # 
        return

    # Construct ethernet frame header and send data
    def send(self, data, eth_type=0x88a4):
        header = struct.pack("!6s6sH", self.dest_mac, self.src_mac, eth_type)

        # print("dest_mac:", self.dest_mac)
        # print("src_mac:", self.src_mac)
    # pack to minimum length
        # if len(data) < 46:
        #     data += "\x00"*(46 - len(data))

        packet = header + data
        # print("sendto!!!!!")

        self.send_sock.sendto(packet, (self.interface, 0))
        return

    def recv(self, bufsize, eth_type=0x0800):
        data = None

        while data == None:
            packet = self.recv_sock.recv(65536)
            # if EthernetSocket.isValid(self, packet, eth_type):
            data = packet[14:]

        return data

    # Checks if the received packet is destined for us
    def isValid(self, packet, desired_type):
        unpacked = struct.unpack("!6s6sH", packet[:14])
        dest_mac = unpacked[0]
        src_mac = unpacked[1]
        eth_type = unpacked[2]

        if eth_type != desired_type:
            return False

        if self.src_mac != dest_mac:
            return False

        # If we are still arping, we don't have a dest_mac set yet
        if desired_type != EthernetSocket.ETH_P_ARP:
            if self.dest_mac != src_mac:
                return False

        return True

    # from stackoverflow.com/questions/2761829/python-get-default-gateway-for-a-local-interface-ip-address-in-linux
    # Gets the address of the default gateway on our interface
    def get_default_gateway(self):
        f = open("/proc/net/route")

        for line in f:
            fields = line.strip().split()
            if fields[0] != self.interface or fields[1] != '00000000' or not int(fields[3], 16) & 2:
                continue

            return fields[2]

    # arps to find the gateway mac address so we can send packets

    # Close sockets

    def close(self):
        self.recv_sock.close()
        self.send_sock.close()
        return
