using Zniffer.Network;
using System;
using System.Text;
using System.ComponentModel;
using System.Windows.Forms;
using System.Collections.ObjectModel;
using PcapDotNet.Packets;
using PcapDotNet.Core;
using PcapDotNet.Packets.IpV4;
using PcapDotNet.Packets.Transport;
using PcapDotNet.Packets.Http;

namespace Zniffer {
    public enum Protocol {
        TCP = 6,
        UDP = 17,
        Unknown = -1
    };
    class Sniffer : Control{

        public ObservableCollection<InterfaceClass> UsedInterfaces = new ObservableCollection<InterfaceClass>();
        public ObservableCollection<PacketDevice> Connections = new ObservableCollection<PacketDevice>();
        public ObservableCollection<BackgroundWorker> Workers = new ObservableCollection<BackgroundWorker>();
        private MainWindow window;

        //
        private void addNewBackgroundWorker(LivePacketDevice adapter) {
            BackgroundWorker backgroundWorker = new BackgroundWorker();
            backgroundWorker.WorkerSupportsCancellation = true;
            
            backgroundWorker.DoWork += (sender, e) => {
                using (PacketCommunicator communicator = adapter.Open(65536, PacketDeviceOpenAttributes.Promiscuous, 1000)) {
                    if (MainWindow.TCP && !MainWindow.UDP)
                        communicator.SetFilter(communicator.CreateFilter("tcp"));
                    if (!MainWindow.TCP && MainWindow.UDP)
                        communicator.SetFilter(communicator.CreateFilter("udp"));
                    if (MainWindow.TCP && MainWindow.UDP)
                        communicator.SetFilter(communicator.CreateFilter("tcp and udp"));
                    communicator.ReceivePackets(0, PacketHandler);
                    while (!backgroundWorker.CancellationPending) {
                        //check if this works
                    }
                    communicator.Break();
                }
            };
            backgroundWorker.RunWorkerAsync();
        }

        private void PacketHandler(Packet packet) {
            IpV4Datagram ip = packet.Ethernet.IpV4;
            TcpDatagram tcp = ip.Tcp;
            UdpDatagram udp = ip.Udp;
            HttpDatagram httpPacket = null;

            //check port range

            //scan datagram

            //if need save file

            


            throw new NotImplementedException();
        }

        private void addNewInterface(InterfaceClass interfaceObj) {
            var list = LivePacketDevice.AllLocalMachine;
            LivePacketDevice adapter = null;
            foreach(var ad in list) {
                if (ad.Addresses[1].Address.ToString().Equals(interfaceObj.Addres)) {//if wrong means interface is only ipv4
                    adapter = ad;
                    break;
                }
            }


            //PacketDevice device = new PacketDevice();

            addNewBackgroundWorker(adapter);



            //ParseData(interfaceObj.byteData, nReceived);
            
        }

        public void removeInterface(InterfaceClass interfaceObj) {
            int index = UsedInterfaces.IndexOf(interfaceObj);
            UsedInterfaces[index].ContinueCapturing = false;

            //stop background workers
            foreach(BackgroundWorker worker in Workers) {
                worker.CancelAsync();
            }
            Workers = new ObservableCollection<BackgroundWorker>();

            //remove connections
        }

        public Sniffer(MainWindow window, ref ObservableCollection<InterfaceClass> UsedInterfaces) {
            this.window = window;
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
            //foreach(var connection in Connections) {
            //    connection.Close();
            //}
            //Connections = new ObservableCollection<Socket>();

            //Callbacks = new ObservableCollection<AsyncCallback>();
        }

        internal void addAllConnections() {
            foreach (var iFace in UsedInterfaces) {
                addNewInterface(iFace);
            }
        }
    }
}
