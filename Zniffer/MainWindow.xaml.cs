using System;
using System.Collections.Generic;
using System.IO;
using System.Management;
using System.Net.NetworkInformation;
using System.Net.Sockets;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading;
using System.Windows;
using System.Windows.Input;
using System.Windows.Interop;
using Trinet.Core.IO.Ntfs;
using Zniffer.Network;

namespace Zniffer {

    /// <summary>
    /// Interaction logic for MainWindow.xaml
    /// </summary>
    public partial class MainWindow : Window {


        #region DEBUG
        public enum DebugLevels {
            NO_DEBUG = 0,
            CRITICAL = 1,
            ERROR = 2,
            WARNING = 3,
            ALL_IO = 4
        };
        public static DebugLevels DEBUG_LEVEL = DebugLevels.CRITICAL;

        public static void log(DebugLevels level, string format, params string[] values) {
            if (DEBUG_LEVEL >= level) {
                Console.Out.WriteLine(string.Format(format, values));
            }
        }
        public static void printLine(string format, params string[] values) {
            Console.Out.WriteLine(string.Format(format, values));
        }
        #endregion



        #region TitleBar buttons
        private void button_Exit_Click(object sender, RoutedEventArgs e) {
            Close();
        }

        private void button_Max_Click(object sender, RoutedEventArgs e) {
            if (WindowState == WindowState.Maximized)
                WindowState = WindowState.Normal;
            else
                WindowState = WindowState.Maximized;
        }

        private void button_Min_Click(object sender, RoutedEventArgs e) {
            WindowState = WindowState.Minimized;
        }
        #endregion



        #region ADS
        private const int GENERIC_WRITE = 1073741824;
        private const int FILE_SHARE_DELETE = 4;
        private const int FILE_SHARE_WRITE = 2;
        private const int FILE_SHARE_READ = 1;
        private const int OPEN_ALWAYS = 4;
        [DllImport("kernel32.dll", SetLastError = true)]
        private static extern IntPtr CreateFile(string lpFileName,
                                                uint dwDesiredAccess,
                                                uint dwShareMode,
                                                IntPtr lpSecurityAttributes,
                                                uint dwCreationDisposition,
                                                uint dwFlagsAndAttributes,
                                                IntPtr hTemplateFile);
        [DllImport("kernel32", SetLastError = true)]
        private static extern bool CloseHandle(IntPtr handle);

        [Obsolete("Do not use")]
        public void tempPisanie() {
            string textToAddToFile = "text to add to file";
            string fileName = "";
            if (fileName != string.Empty) {
                FileInfo fileInfo = new FileInfo(fileName);
                int len = 0;
                len = textToAddToFile.Length * sizeof(char);
                byte[] bytes = new byte[textToAddToFile.Length * sizeof(char)];
                Buffer.BlockCopy(textToAddToFile.ToCharArray(), 0, bytes, 0, bytes.Length);

                using (BinaryWriter bw = new BinaryWriter(File.Open(fileName, FileMode.Append), Encoding.UTF8)) {
                    bw.Write(bytes);
                }

                uint crc = 123456;
                if (!fileInfo.AlternateDataStreamExists("crc")) {
                    var stream = CreateFile(
                    fileName + ":crc",
                    GENERIC_WRITE,
                    FILE_SHARE_WRITE,
                    IntPtr.Zero,
                    OPEN_ALWAYS,
                    0,
                    IntPtr.Zero);
                    if (stream != IntPtr.Zero)
                        CloseHandle(stream);
                }
                FileStream fs = fileInfo.GetAlternateDataStream("crc").OpenWrite();
                fs.Write(BitConverter.GetBytes(crc), 0, 4);
                fs.Close();
            }
        }

        [Obsolete("Do not use")]
        public void tempCzytanie() {
            string pathToFile = "";
            bool toReturn = false;
            uint crcFromFile;
            byte[] arr = new byte[4];
            FileInfo fileInfo = new FileInfo(pathToFile);
            uint crc = 123456;
            if (fileInfo.AlternateDataStreamExists("crc")) {
                foreach (AlternateDataStreamInfo stream in fileInfo.ListAlternateDataStreams()) {

                }
                using (FileStream fs = fileInfo.GetAlternateDataStream("crc").OpenRead()) {
                    fs.Read(arr, 0, 4);
                }
                crcFromFile = BitConverter.ToUInt32(arr, 0);
                if (crcFromFile == crc)
                    toReturn = true;
                else toReturn = false;
            }
            //return toReturn;
        }


        #endregion



        #region Clipboard

        #region DataFromats

        string[] formatsAll = new string[]
        {
            DataFormats.Bitmap,
            DataFormats.CommaSeparatedValue,
            DataFormats.Dib,
            DataFormats.Dif,
            DataFormats.EnhancedMetafile,
            DataFormats.FileDrop,
            DataFormats.Html,
            DataFormats.Locale,
            DataFormats.MetafilePicture,
            DataFormats.OemText,
            DataFormats.Palette,
            DataFormats.PenData,
            DataFormats.Riff,
            DataFormats.Rtf,
            DataFormats.Serializable,
            DataFormats.StringFormat,
            DataFormats.SymbolicLink,
            DataFormats.Text,
            DataFormats.Tiff,
            DataFormats.UnicodeText,
            DataFormats.WaveAudio
        };

        #endregion

        [DllImport("User32.dll", CharSet = CharSet.Auto)]
        public static extern IntPtr SetClipboardViewer(IntPtr hWndNewViewer);
        [DllImport("User32.dll", CharSet = CharSet.Auto)]
        public static extern bool ChangeClipboardChain(
            IntPtr hWndRemove,  // handle to window to remove
            IntPtr hWndNewNext  // handle to next window
            );

        IntPtr clipboardViewerNext;

        private static IntPtr WndProc(IntPtr hwnd, int msg, IntPtr wParam, IntPtr lParam, ref bool handled) {
            if (msg == 0x0308) {
                printLine("Data retrived from clipboard");

                IDataObject iData = new DataObject();

                try {
                    iData = Clipboard.GetDataObject();
                } catch (ExternalException externEx) {
                    printLine("InteropServices.ExternalException: {0}", externEx.Message);
                    return IntPtr.Zero; ;
                } catch (Exception ex) {
                    return IntPtr.Zero; ;
                }

                if (iData.GetDataPresent(DataFormats.Rtf)) {
                    printLine((string)iData.GetData(DataFormats.Rtf));

                }
                if (iData.GetDataPresent(DataFormats.Text)) {
                    printLine((string)iData.GetData(DataFormats.Text));

                } else {
                    printLine("(cannot display this format)");
                }
            }
            return IntPtr.Zero;
        }
        #endregion


        public Thread keyLogger;
        public Dictionary<string, string> avaliableDrives = new Dictionary<string, string>();
        public Dictionary<string, string> avaliableNetworkAdapters = new Dictionary<string, string>();


        public MainWindow() {
            InitializeComponent();
        }


        public static void printTextInConsole(string s) {

            if (s.Substring(0, 1).Equals("<") && s.Substring(s.Length - 1, 1).Equals(">")) {//znaki specjalne
                ;
            } else {//zwykle znaki

                Console.Out.WriteLine(s);
            }
        }

        private void newDeviceDetectedEventArived(object sender, EventArrivedEventArgs e) {
            Console.Out.WriteLine(e.NewEvent.Properties["DriveName"].Value.ToString() + " inserted");
        }



        private void Window_SourceInitialized(object sender, EventArgs e) {
            //sniffer
            Sniffer snf = new Sniffer();


            //nasluchiwanie schowka
            clipboardViewerNext = SetClipboardViewer(new WindowInteropHelper(this).Handle);
            HwndSource source = HwndSource.FromHwnd(new WindowInteropHelper(this).Handle);
            source.AddHook(new HwndSourceHook(WndProc));



            //sprawdzenie systemów plików // dodac sprawdzanie czy jest ntfs
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
                avaliableDrives.Add(d.Name, d.DriveFormat);
            }

            //sprawdzenie czy jest karta graficzna
            ManagementObjectSearcher searcher = new ManagementObjectSearcher("SELECT * FROM Win32_VideoController");
            foreach (ManagementObject mo in searcher.Get()) {
                //PropertyData currentBitsPerPixel = mo.Properties["CurrentBitsPerPixel"];
                PropertyData description = mo.Properties["Description"];
                /*if (currentBitsPerPixel != null && description != null) {
                    if (currentBitsPerPixel.Value != null)
                        ;
                }*/
                printLine(description.Value.ToString());
            }

            //sprawdzenie ilosci ramu
            double amoutOfRam = new Microsoft.VisualBasic.Devices.ComputerInfo().TotalPhysicalMemory;
            double amountOfMBOfRam = amoutOfRam / 1024 / 1024;
            printLine("" + amountOfMBOfRam);

            //wykrywanie nowych polaczen sieciowych itp
            NetworkChange.NetworkAddressChanged += new NetworkAddressChangedEventHandler(AddressChangedCallback);


            //sprawdzenie ile jest kart sieciowych
            NetworkInterface[] adapters = NetworkInterface.GetAllNetworkInterfaces();
            foreach (NetworkInterface adapter in adapters) {
                string ipAddrList = string.Empty;
                IPInterfaceProperties properties = adapter.GetIPProperties();
                printLine(adapter.Description);
                printLine("  DNS suffix .............................. : {0}", properties.DnsSuffix);
                printLine("  DNS enabled ............................. : {0}", properties.IsDnsEnabled.ToString());
                printLine("  Dynamically configured DNS .............. : {0}", properties.IsDynamicDnsEnabled.ToString());

                if (adapter.NetworkInterfaceType == NetworkInterfaceType.Ethernet && adapter.OperationalStatus == OperationalStatus.Up) {
                    foreach (UnicastIPAddressInformation ip in adapter.GetIPProperties().UnicastAddresses)
                        if (ip.Address.AddressFamily == AddressFamily.InterNetwork)
                            printLine("Ip Addresses " + ip.Address.ToString());
                }
                printLine("\n");
            }

            printLine("\n");
            printLine("User Domain Name: " + Environment.UserDomainName);
            printLine("Machine Name: " + Environment.MachineName);
            printLine("User Name " + Environment.UserName);



            //wykrywanie podlaczenia pamieci typu flash
            ManagementEventWatcher watcher = new ManagementEventWatcher();
            WqlEventQuery query = new WqlEventQuery("SELECT * FROM Win32_VolumeChangeEvent WHERE EventType = 2");
            watcher.EventArrived += new EventArrivedEventHandler(newDeviceDetectedEventArived);
            watcher.Query = query;
            watcher.Start();
            //watcher.WaitForNextEvent();

            //powolanie keyloggera
            var obj = new KeyLogger();
            obj.RaiseKeyCapturedEvent += new KeyLogger.keyCaptured(printTextInConsole);


        }

        static void AddressChangedCallback(object sender, EventArgs e) {
            NetworkInterface[] adapters = NetworkInterface.GetAllNetworkInterfaces();
            foreach (NetworkInterface n in adapters) {
                foreach (UnicastIPAddressInformation ip in n.GetIPProperties().UnicastAddresses)
                    if (ip.Address.AddressFamily == AddressFamily.InterNetwork)
                        printLine("   {0} is {1}", n.Name, n.OperationalStatus.ToString());
            }
        }

        private void Window_Closing(object sender, System.ComponentModel.CancelEventArgs e) {
            ChangeClipboardChain(new WindowInteropHelper(this).Handle, clipboardViewerNext);
        }

        private void NetworkOnMouseWheel(object sender, System.Windows.Input.MouseWheelEventArgs e) {
            bool handle = (Keyboard.Modifiers & ModifierKeys.Control) > 0;
            if (!handle)
                return;
            if (e.Delta > 0 && NetworkTextBlock.FontSize < 40.0)
                NetworkTextBlock.FontSize++;

            if (e.Delta < 0 && NetworkTextBlock.FontSize > 8.0)
                NetworkTextBlock.FontSize--;
        }

        private void FilesTextBlock_MouseWheel(object sender, MouseWheelEventArgs e) {
            bool handle = (Keyboard.Modifiers & ModifierKeys.Control) > 0;
            if (!handle)
                return;
            if (e.Delta > 0 && FilesTextBlock.FontSize < 40.0)
                FilesTextBlock.FontSize++;

            if (e.Delta < 0 && FilesTextBlock.FontSize > 8.0)
                FilesTextBlock.FontSize--;
        }

        private void ClipboardTextBlock_MouseWheel(object sender, MouseWheelEventArgs e) {
            bool handle = (Keyboard.Modifiers & ModifierKeys.Control) > 0;
            if (!handle)
                return;
            if (e.Delta > 0 && ClipboardTextBlock.FontSize < 40.0)
                ClipboardTextBlock.FontSize++;

            if (e.Delta < 0 && ClipboardTextBlock.FontSize > 8.0)
                ClipboardTextBlock.FontSize--;
        }
    }
}
