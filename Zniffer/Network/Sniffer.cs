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
using System.Collections.ObjectModel;
using PcapDotNet.Packets;
using PcapDotNet.Core;

namespace Zniffer {
    public enum Protocol {
        TCP = 6,
        UDP = 17,
        Unknown = -1
    };
    class Sniffer : Control{

        public ObservableCollection<InterfaceClass> UsedInterfaces = new ObservableCollection<InterfaceClass>();
        public ObservableCollection<Socket> Connections = new ObservableCollection<Socket>();
        public ObservableCollection<AsyncCallback> Callbacks = new ObservableCollection<AsyncCallback>();

        //

        private void addNewInterface(InterfaceClass interfaceObj) {
            Socket socket = new Socket(System.Net.Sockets.AddressFamily.InterNetwork, SocketType.Raw, ProtocolType.IP);
            Connections.Add(socket);

            socket.Bind(new IPEndPoint(IPAddress.Parse(interfaceObj.Addres), 0));

            socket.SetSocketOption(SocketOptionLevel.IP, SocketOptionName.HeaderIncluded, true);           //Applies only to IP packets Set the include the header option to true

            byte[] byTrue = new byte[4] { 1, 0, 0, 0 };
            byte[] byOut = new byte[4] { 1, 0, 0, 0 }; //Capture outgoing packets

            //Socket.IOControl is analogous to the WSAIoctl method of Winsock 2 Equivalent to SIO_RCVALL constant of Winsock 2
            socket.IOControl(IOControlCode.ReceiveAll, byTrue, byOut);


            //socket.ReceiveAsync();

            //Start receiving the packets asynchronously
            AsyncCallback callback = null;
            callback = ar => {
                try {
                    int nReceived = socket.EndReceive(ar);

                    //Analyze the bytes received...
                    ParseData(interfaceObj.byteData, nReceived);
                    //
                    if (interfaceObj.ContinueCapturing) {
                        interfaceObj.byteData = new byte[4096];

                        //Another call to BeginReceive so that we continue to receive the incoming packets
                        socket.BeginReceive(interfaceObj.byteData, 0, interfaceObj.byteData.Length, SocketFlags.None, new AsyncCallback(callback), null);
                    }
                }
                catch (ObjectDisposedException) {
                }
                catch (Exception ex) {
                    MessageBox.Show(ex.Message, "Sniffer", MessageBoxButtons.OK, MessageBoxIcon.Error);
                }
            };

            Callbacks.Add(callback);

            socket.BeginReceive(interfaceObj.byteData, 0, interfaceObj.byteData.Length, SocketFlags.None, new AsyncCallback(callback), null);
        }

        public void removeInterface(InterfaceClass interfaceObj) {
            int index = UsedInterfaces.IndexOf(interfaceObj);
            UsedInterfaces[index].ContinueCapturing = false;
        }

        public Sniffer(ref ObservableCollection<InterfaceClass> UsedInterfaces) {
            this.UsedInterfaces = UsedInterfaces;
            UsedInterfaces.CollectionChanged += UsedInterfaces_CollectionChanged;
            //this.UsedInterfaces = UsedInterfaces;

            /*
            //list interfaces
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
            */
        }

        private void UsedInterfaces_CollectionChanged(object sender, System.Collections.Specialized.NotifyCollectionChangedEventArgs e) {
            //Only if main does not overrides this event

        }

                /*
        private void OnReceive(IAsyncResult ar) {
           try {
               int nReceived = mainSocket.EndReceive(ar);

               //Analyze the bytes received...



               ParseData(byteData, nReceived);

               //
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
        }*/

        private void ParseData(byte[] byteData, int nReceived) {
            //Since all protocol packets are encapsulated in the IP datagram
            //so we start by parsing the IP header and see what protocol data
            //is being carried by it
            IPHeader ipHeader = new IPHeader(byteData, nReceived);

            //Now according to the protocol being carried by the IP datagram we parse 
            //the data field of the datagram
            switch (ipHeader.ProtocolType) {
                case Protocol.TCP:
                    TCPHeader tcpHeader = new TCPHeader(ipHeader.Data, ipHeader.MessageLength);//IPHeader.Data stores the data being carried by the IP datagram Length of the data field   

                    //If the port is equal to 53 then the underlying protocol is DNS
                    //Note: DNS can use either TCP or UDP thats why the check is done twice
                    if (tcpHeader.DestinationPort == "53" || tcpHeader.SourcePort == "53") {
                        DNSHeader dnsHeader = new DNSHeader(tcpHeader.Data, (int)tcpHeader.MessageLength);
                        Console.Write("DNS/");
                    }
                    Console.Write(tcpHeader.DestinationPort);
                    Console.WriteLine("/" + ipHeader.ProtocolType + "/" + ipHeader.SourceAddress.ToString() + "-" + ipHeader.DestinationAddress.ToString()
                + "\n\t:" + Encoding.Default.GetString(tcpHeader.Data));
                    break;

                case Protocol.UDP:
                    UDPHeader udpHeader = new UDPHeader(ipHeader.Data,              //IPHeader.Data stores the data being 
                                                                                    //carried by the IP datagram
                                                       (int)ipHeader.MessageLength);//Length of the data field                    

                    //If the port is equal to 53 then the underlying protocol is DNS
                    //Note: DNS can use either TCP or UDP thats why the check is done twice
                    if (udpHeader.DestinationPort == "53" || udpHeader.SourcePort == "53") {
                        Console.Write("DNS/");
                        DNSHeader dnsHeader = new DNSHeader(udpHeader.Data,
                                                           //Length of UDP header is always eight bytes so we subtract that out of the total 
                                                           //length to find the length of the data
                                                           Convert.ToInt32(udpHeader.Length) - 8);
                    }
                    Console.Write(udpHeader.DestinationPort);
                    Console.WriteLine("/" + ipHeader.ProtocolType + "/" + ipHeader.SourceAddress.ToString() + "-" + ipHeader.DestinationAddress.ToString()
                + "\n\t:" + Encoding.UTF8.GetString(udpHeader.Data));
                    break;
                case Protocol.Unknown:
                    break;
            }
            
        }

        internal void removeAllConnections() {
            foreach(var connection in Connections) {
                connection.Close();
            }
            Connections = new ObservableCollection<Socket>();

            Callbacks = new ObservableCollection<AsyncCallback>();
        }

        internal void addAllConnections() {
            foreach (var iFace in UsedInterfaces) {
                addNewInterface(iFace);
            }
        }
    }
}
