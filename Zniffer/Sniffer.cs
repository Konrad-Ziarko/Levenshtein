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

            
        }
    }
}
