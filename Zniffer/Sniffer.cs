using System.IO;
using System.Net.Sockets;
using System.Threading;
using Zniffer.Network;
using System.Management;
using System;
using System.Net.NetworkInformation;
using System.Runtime.InteropServices;
using Trinet.Core.IO.Ntfs;
using System.Text;
using System.Collections.Generic;
using System.ComponentModel;
using System.Windows.Forms;
using System.Windows.Interop;
using CustomExtensions;
using System.Net;

namespace Zniffer {
    public enum Protocol {
        TCP = 6,
        UDP = 17,
        Unknown = -1
    };
    class Sniffer : Control{

       
        //
        private Socket mainSocket;                          //The socket which captures all incoming packets
        private byte[] byteData = new byte[4096];
        private bool bContinueCapturing = false;            //A flag to check if packets are to be captured or not


        public Sniffer() {


            Console.Out.WriteLine();

            string strIP = null;

            IPHostEntry HosyEntry = Dns.GetHostEntry((Dns.GetHostName()));
            if (HosyEntry.AddressList.Length > 0) {
                foreach (IPAddress ip in HosyEntry.AddressList) {
                    if (ip.AddressFamily == AddressFamily.InterNetwork) {
                        strIP = ip.ToString();
                        Console.Out.WriteLine(strIP);
                        //cmbInterfaces.Items.Add(strIP);
                    }

                }
            }


            mainSocket = new Socket(AddressFamily.InterNetwork, SocketType.Raw, ProtocolType.IP);

            //Bind the socket to the selected IP address
            mainSocket.Bind(new IPEndPoint(IPAddress.Parse("192.168.0.143"), 0));

            //Set the socket  options
            mainSocket.SetSocketOption(SocketOptionLevel.IP,            //Applies only to IP packets
                                       SocketOptionName.HeaderIncluded, //Set the include the header
                                       true);                           //option to true

            byte[] byTrue = new byte[4] { 1, 0, 0, 0 };
            byte[] byOut = new byte[4] { 1, 0, 0, 0 }; //Capture outgoing packets

            //Socket.IOControl is analogous to the WSAIoctl method of Winsock 2
            mainSocket.IOControl(IOControlCode.ReceiveAll,              //Equivalent to SIO_RCVALL constant
                                                                        //of Winsock 2
                                 byTrue,
                                 byOut);

            //Start receiving the packets asynchronously
            mainSocket.BeginReceive(byteData, 0, byteData.Length, SocketFlags.None,
                new AsyncCallback(OnReceive), null);


        }

        private void OnReceive(IAsyncResult ar) {
            try {
                int nReceived = mainSocket.EndReceive(ar);

                //Analyze the bytes received...

                ParseData(byteData, nReceived);

                if (bContinueCapturing) {
                    byteData = new byte[4096];

                    //Another call to BeginReceive so that we continue to receive the incoming
                    //packets
                    mainSocket.BeginReceive(byteData, 0, byteData.Length, SocketFlags.None,
                        new AsyncCallback(OnReceive), null);
                }
            }
            catch (ObjectDisposedException) {
            }
            catch (Exception ex) {
                MessageBox.Show(ex.Message, "MJsniffer", MessageBoxButtons.OK, MessageBoxIcon.Error);
            }
        }

        private void ParseData(byte[] byteData, int nReceived) {
            //Since all protocol packets are encapsulated in the IP datagram
            //so we start by parsing the IP header and see what protocol data
            //is being carried by it
            IPHeader ipHeader = new IPHeader(byteData, nReceived);

            //Now according to the protocol being carried by the IP datagram we parse 
            //the data field of the datagram
            switch (ipHeader.ProtocolType) {
                case Protocol.TCP:

                    TCPHeader tcpHeader = new TCPHeader(ipHeader.Data,              //IPHeader.Data stores the data being 
                                                                                    //carried by the IP datagram
                                                        ipHeader.MessageLength);//Length of the data field                    

                    //If the port is equal to 53 then the underlying protocol is DNS
                    //Note: DNS can use either TCP or UDP thats why the check is done twice
                    if (tcpHeader.DestinationPort == "53" || tcpHeader.SourcePort == "53") {
                        DNSHeader dnsHeader = new DNSHeader(tcpHeader.Data, (int)tcpHeader.MessageLength);
                    }

                    break;

                case Protocol.UDP:

                    UDPHeader udpHeader = new UDPHeader(ipHeader.Data,              //IPHeader.Data stores the data being 
                                                                                    //carried by the IP datagram
                                                       (int)ipHeader.MessageLength);//Length of the data field                    

                    //If the port is equal to 53 then the underlying protocol is DNS
                    //Note: DNS can use either TCP or UDP thats why the check is done twice
                    if (udpHeader.DestinationPort == "53" || udpHeader.SourcePort == "53") {

                        DNSHeader dnsHeader = new DNSHeader(udpHeader.Data,
                                                           //Length of UDP header is always eight bytes so we subtract that out of the total 
                                                           //length to find the length of the data
                                                           Convert.ToInt32(udpHeader.Length) - 8);
                    }

                    break;

                case Protocol.Unknown:
                    break;
            }

            Console.WriteLine(ipHeader.SourceAddress.ToString() + "-" + ipHeader.DestinationAddress.ToString());

        }
    }
}
