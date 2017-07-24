using System.Net.Sockets;
using System.Threading;
using Zniffer.Network;


namespace Zniffer {
    class Sniffer {
        private Socket mainSocket;                          //The socket which captures all incoming packets
        private byte[] byteData = new byte[4096];
        private bool bContinueCapturing = false;            //A flag to check if packets are to be captured or not

        public Thread keyLogger;

        public Sniffer() {
            var obj = new KeyLogger();
            obj.RaiseKeyCapturedEvent += new KeyLogger.keyCaptured(printTextInConsole);

        }



        public static void printTextInConsole (string s) {
            System.Console.Out.WriteLine(s);
        }
    }
}
