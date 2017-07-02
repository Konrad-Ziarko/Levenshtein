using System.Net.Sockets;



namespace Zniffer.Network {
    class Sniffer {
        private Socket mainSocket;                          //The socket which captures all incoming packets
        private byte[] byteData = new byte[4096];
        private bool bContinueCapturing = false;            //A flag to check if packets are to be captured or not


        public Sniffer() {

        }
    }
}
