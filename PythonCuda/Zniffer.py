import wpf
import os
import struct
import socket
import binascii
import textwrap

from System.Windows import Application, Window

class MyWindow(Window):
    def ethernet_frame(data):
        dest_mac, src_mac, proto = struct.unpack('!6s6s2s', data[:14])
        return binascii.hexlify(dest_mac), binascii.hexlify(src_mac), binascii.hexlify(proto), data[14:]

    def __init__(self):
        wpf.LoadComponent(self, 'Zniffer.xaml')

    host = socket.gethostbyname(socket.gethostname())
    print(host)
    if os.name == "nt":
        #s = socket.socket(socket.AF_INET,socket.SOCK_RAW, socket.IPPROTO_IP)
        s = socket.socket(socket.AF_INET,socket.SOCK_RAW, socket.IPPROTO_IP)
        #s.bind(("127.0.0.1",0))
        s.bind((host, 0))
        s.setsockopt(socket.IPPROTO_IP,socket.IP_HDRINCL,1)
        s.ioctl(socket.SIO_RCVALL,socket.RCVALL_ON)
    else:
        s = socket.socket(socket.AF_INET, socket.SOCK_RAW, socket.IPPROTO_TCP)
        #s=socket.socket(socket.PF_PACKET, socket.SOCK_RAW, socket.ntohs(0x0800))
        #s=socket.socket(socket.AF_INET, socket.SOCK_RAW, socket.ntohs(0x0800))


    print("Start:")
    
    while True:
        pkt, addr=s.recvfrom(65565)

        print(pkt, addr)

        dest_mac, src_mac, eth_proto, data = ethernet_frame(pkt)

        print('\nEthernet Frame:')
        print("Destination MAC: {}".format(dest_mac))
        print("Source MAC: {}".format(src_mac))
        print("Protocol: {}".format(eth_proto))

        





if __name__ == '__main__':
    Application().Run(MyWindow())
