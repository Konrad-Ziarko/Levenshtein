using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.IO;
using System.Management;
using System.Net.NetworkInformation;
using System.Net.Sockets;
using System.Runtime.InteropServices;
using System.Text;
using System.Windows;
using System.Windows.Input;
using System.Windows.Interop;
using Trinet.Core.IO.Ntfs;

namespace Zniffer {

    /// <summary>
    /// Interaction logic for MainWindow.xaml
    /// </summary>
    public partial class MainWindow : Window {

        #region Networking
        public static Dictionary<string, string> AvaliableNetworkAdapters = new Dictionary<string, string>();
        public ObservableCollection<InterfaceClass> UsedInterfaces = new ObservableCollection<InterfaceClass>();
        public ObservableCollection<InterfaceClass> AvaliableInterfaces = new ObservableCollection<InterfaceClass>();


        public ObservableCollection<InterfaceClass> UsedFaces {
            get {
                return UsedInterfaces;
            }
        }
        public ObservableCollection<InterfaceClass> AvaliableFaces {
            get {
                return AvaliableInterfaces;
            }
        }

        BaseWindow networkSettingsWindow, fileExtensionsWindow;


        //Not used
        public void SearchForNewNetworkInterfaces() {
            //szukaj nowych interfejsow co pewien interwal
            // jesli okno networksettings jest otwarte wyrzuci wyjatek (dostep z innego watku))

            if (networkSettingsWindow != null) {
                try {
                    //networkSettingsWindow.addAvaliableInterface(new InterfaceClass("1232132132", ""));
                }
                catch {
                    AvaliableFaces.Add(new InterfaceClass("1232132132", ""));
                }
            }
            else {
                AvaliableFaces.Add(new InterfaceClass("1232132132", ""));
            }
        }


        #endregion

        #region File Extension

        public ObservableCollection<FileExtensionClass> UsedExtensions = new ObservableCollection<FileExtensionClass>();
        public ObservableCollection<FileExtensionClass> AvaliableExtensions = new ObservableCollection<FileExtensionClass>();
        public ObservableCollection<FileExtensionClass> UsedExt {
            get {
                return UsedExtensions;
            }
        }
        public ObservableCollection<FileExtensionClass> AvaliableExt {
            get {
                return AvaliableExtensions;
            }
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
        public static extern bool ChangeClipboardChain(IntPtr hWndRemove, IntPtr hWndNewNext);

        IntPtr clipboardViewerNext;

        private static IntPtr WndProc(IntPtr hwnd, int msg, IntPtr wParam, IntPtr lParam, ref bool handled) {
            if (msg == 0x0308) {
                //printLine("Data retrived from clipboard");
                //wyciąganie danych z obrazów i dźwięków
                IDataObject iData = new DataObject();

                try {
                    iData = Clipboard.GetDataObject();
                }
                catch (ExternalException externEx) {
                    Console.Out.WriteLine("InteropServices.ExternalException: {0}", externEx.Message);

                    //TODO zrobić obsługę schowka
                    //print screen to też zmiana schowka

                    return IntPtr.Zero; ;
                }
                catch (Exception) {
                    return IntPtr.Zero; ;
                }
                if (iData.GetDataPresent(DataFormats.Rtf)) {
                    Console.Out.WriteLine((string)iData.GetData(DataFormats.Rtf));

                }
                else if (iData.GetDataPresent(DataFormats.Text)) {
                    Console.Out.WriteLine((string)iData.GetData(DataFormats.Text));

                }
                else {
                    Console.Out.WriteLine("(cannot display this format)");
                }
            }
            return IntPtr.Zero;
        }
        #endregion

        public static System.Timers.Timer resetStringTimer = new System.Timers.Timer(5000);//5sec reset time

        public Dictionary<string, string> avaliableDrives = new Dictionary<string, string>();




        public static string loggedKeyString = "";
        public static long cursorPosition = 0;

        public MainWindow() {
            InitializeComponent();
            this.DataContext = this;

            if (Properties.Settings.Default.UsedExtensions == null)
                Properties.Settings.Default.UsedExtensions = new System.Collections.Specialized.StringCollection();
            if (Properties.Settings.Default.AvaliableExtensions == null)
                Properties.Settings.Default.AvaliableExtensions = new System.Collections.Specialized.StringCollection();

            foreach (string ext in Properties.Settings.Default.AvaliableExtensions) {
                AvaliableExt.Add(new FileExtensionClass(ext));
            }
            foreach (string ext in Properties.Settings.Default.UsedExtensions) {
                UsedExt.Add(new FileExtensionClass(ext));
            }

            resetStringTimer.Elapsed += new System.Timers.ElapsedEventHandler(OnResetTimerEvent);
        }

        public void OnResetTimerEvent(object source, System.Timers.ElapsedEventArgs e) {
            cursorPosition = 0;
            loggedKeyString = "";
            
        }

        public static void keyCapturedHandle(string s) {

            if (s.Substring(0, 1).Equals("<") && s.Substring(s.Length - 1, 1).Equals(">")) {//special characters
                s = s.Substring(1, s.Length - 2);
                if (s.Equals("Backspace")) {
                    resetStringTimer.Stop();

                    loggedKeyString = loggedKeyString.Remove(loggedKeyString.Length - 1);//wyjatek jesli backspace wiecej niz bylo nacisniete
                    cursorPosition--;
                }
                else if (s.Equals("Left")) {
                    resetStringTimer.Stop();

                    if (cursorPosition > 0)
                        cursorPosition--;
                }
                else if (s.Equals("Right")) {
                    resetStringTimer.Stop();

                    if (cursorPosition < loggedKeyString.Length)
                        cursorPosition++;
                }
            }
            else if (s.Substring(0, 1).Equals("[") && s.Substring(s.Length - 1, 1).Equals("]")) {//active window changed
                resetStringTimer.Stop();

            }
            else {//normal characters
                resetStringTimer.Stop();

                loggedKeyString += s;
                cursorPosition++;

                if (s.Equals("\n")) {//user returned(ended) string
                    cursorPosition = 0;
                    loggedKeyString = "";
                }
            }

            //Console.Out.WriteLine(Searcher.extractPhrase(loggedKeyString, _SearchPhrase));

            resetStringTimer.Start();
        }

        //discovering new drives
        private async void newDeviceDetectedEventArived(object sender, EventArrivedEventArgs e) {

            //TODO async scan files on newly attached devices (if ntfs +ADS)

            Console.Out.WriteLine(e.NewEvent.Properties["DriveName"].Value.ToString() + " inserted");

            foreach (DriveInfo d in DriveInfo.GetDrives()) {
                if (d.Name.Equals(e.NewEvent.Properties["DriveName"].Value.ToString() + "\\")) {
                    List<string> directories = Searcher.GetDirectories(d.Name);

                    foreach (string directory in directories) {
                        Console.Out.WriteLine(directory);

                        List<string> files = Searcher.GetFiles(directory);

                        foreach (string file in files) {
                            Console.Out.WriteLine(file);
                            //Console.Out.WriteLine(File.ReadAllText(file));
                            try {
                                string arr = await Searcher.ReadTextAsync(file);
                                //foreach(string str in File.ReadLines(file))
                                Console.Out.WriteLine(arr);

                            }
                            catch (UnauthorizedAccessException) {
                                Console.Out.WriteLine("Cannot access:" + file);
                            }
                            if (d.DriveFormat.Equals("NTFS")) {
                                //search for ads
                            }
                        }
                    }
                }

            }
        }



        public void Window_SourceInitialized(object sender, EventArgs e) {
            //TODO implement sniffer
            Sniffer snf = new Sniffer();



            //attach to clipboard
            clipboardViewerNext = SetClipboardViewer(new WindowInteropHelper(this).Handle);
            HwndSource source = HwndSource.FromHwnd(new WindowInteropHelper(this).Handle);
            source.AddHook(new HwndSourceHook(WndProc));



            //check file system // if ntfs look for ADSs
            foreach (DriveInfo d in DriveInfo.GetDrives()) {
                Console.Out.WriteLine("Drive {0}", d.Name);
                Console.Out.WriteLine("  Drive type: {0}", d.DriveType.ToString());
                if (d.IsReady == true) {
                    Console.Out.WriteLine("  Volume label: {0}", d.VolumeLabel);
                    Console.Out.WriteLine("  File system: {0}", d.DriveFormat);
                    Console.Out.WriteLine("  Available space to current user:{0, 15} bytes", d.AvailableFreeSpace.ToString());
                    Console.Out.WriteLine("  Total available space:          {0, 15} bytes", d.TotalFreeSpace.ToString());
                    Console.Out.WriteLine("  Total size of drive:            {0, 15} bytes ", d.TotalSize.ToString());
                }
                try {
                    avaliableDrives.Add(d.Name, d.DriveFormat);
                }
                catch (ArgumentException) {
                    //pojawił się dysk o tej samej literce
                }

            }

            //check for GPU
            ManagementObjectSearcher searcher = new ManagementObjectSearcher("SELECT * FROM Win32_VideoController");
            foreach (ManagementObject mo in searcher.Get()) {
                //PropertyData currentBitsPerPixel = mo.Properties["CurrentBitsPerPixel"];
                PropertyData description = mo.Properties["Description"];
                /*if (currentBitsPerPixel != null && description != null) {
                    if (currentBitsPerPixel.Value != null)
                        ;
                }*/
                Console.Out.WriteLine(description.Value.ToString());
            }

            //check for the amount of ram
            double amoutOfRam = new Microsoft.VisualBasic.Devices.ComputerInfo().TotalPhysicalMemory;
            double amountOfMBOfRam = amoutOfRam / 1024 / 1024;
            Console.Out.WriteLine("" + amountOfMBOfRam);


            //detect new network connections/interfaces
            NetworkChange.NetworkAddressChanged += new NetworkAddressChangedEventHandler(AddressChangedCallback);

            //look for network adapters
            NetworkInterface[] adapters = NetworkInterface.GetAllNetworkInterfaces();
            //timer do szukania nowych interfejsów sieciowych albo nasłuchiwanie eventów o nowym interfejsie
            foreach (NetworkInterface adapter in adapters) {
                string ipAddrList = string.Empty;
                IPInterfaceProperties properties = adapter.GetIPProperties();
                Console.Out.WriteLine(adapter.Description);
                Console.Out.WriteLine("  DNS suffix .............................. : {0}", properties.DnsSuffix);
                Console.Out.WriteLine("  DNS enabled ............................. : {0}", properties.IsDnsEnabled.ToString());
                Console.Out.WriteLine("  Dynamically configured DNS .............. : {0}", properties.IsDynamicDnsEnabled.ToString());

                if (adapter.NetworkInterfaceType == NetworkInterfaceType.Ethernet && adapter.OperationalStatus == OperationalStatus.Up) {
                    foreach (UnicastIPAddressInformation ip in adapter.GetIPProperties().UnicastAddresses)
                        if (ip.Address.AddressFamily == AddressFamily.InterNetwork) {
                            Console.Out.WriteLine("Ip Addresses " + ip.Address.ToString());
                            AvaliableInterfaces.Add(new InterfaceClass(ip.Address.ToString(), ""));
                        }
                }
                Console.Out.WriteLine("\n");
            }


            //Print more info
            Console.Out.WriteLine("\n");
            Console.Out.WriteLine("User Domain Name: " + Environment.UserDomainName);
            Console.Out.WriteLine("Machine Name: " + Environment.MachineName);
            Console.Out.WriteLine("User Name " + Environment.UserName);



            //detect flash memory
            ManagementEventWatcher watcher = new ManagementEventWatcher();
            WqlEventQuery query = new WqlEventQuery("SELECT * FROM Win32_VolumeChangeEvent WHERE EventType = 2");
            watcher.EventArrived += new EventArrivedEventHandler(newDeviceDetectedEventArived);
            watcher.Query = query;
            watcher.Start();
            //watcher.WaitForNextEvent();

            //run keylogger
            var obj = new KeyLogger();
            obj.RaiseKeyCapturedEvent += new KeyLogger.keyCaptured(keyCapturedHandle);


        }

        static void AddressChangedCallback(object sender, EventArgs e) {
            NetworkInterface[] adapters = NetworkInterface.GetAllNetworkInterfaces();
            foreach (NetworkInterface n in adapters) {
                foreach (UnicastIPAddressInformation ip in n.GetIPProperties().UnicastAddresses)
                    if (ip.Address.AddressFamily == AddressFamily.InterNetwork)
                        Console.Out.WriteLine("   {0} is {1}", n.Name, n.OperationalStatus.ToString());
            }
        }

        private void Window_Closing(object sender, System.ComponentModel.CancelEventArgs e) {
            ChangeClipboardChain(new WindowInteropHelper(this).Handle, clipboardViewerNext);

            Properties.Settings.Default.Save();
        }


        #region MouseWheelFont

        private void NetworkOnMouseWheel(object sender, MouseWheelEventArgs e) {
            bool handle = (Keyboard.Modifiers & ModifierKeys.Control) > 0;
            if (!handle)
                return;
            if (e.Delta > 0 && NetworkTextBlock.FontSize < 80.0)
                NetworkTextBlock.FontSize++;

            if (e.Delta < 0 && NetworkTextBlock.FontSize > 12.0)
                NetworkTextBlock.FontSize--;
        }

        private void FilesTextBlock_MouseWheel(object sender, MouseWheelEventArgs e) {
            bool handle = (Keyboard.Modifiers & ModifierKeys.Control) > 0;
            if (!handle)
                return;
            if (e.Delta > 0 && FilesTextBlock.FontSize < 80.0)
                FilesTextBlock.FontSize++;

            if (e.Delta < 0 && FilesTextBlock.FontSize > 12.0)
                FilesTextBlock.FontSize--;
        }

        private void ClipboardTextBlock_MouseWheel(object sender, MouseWheelEventArgs e) {
            bool handle = (Keyboard.Modifiers & ModifierKeys.Control) > 0;
            if (!handle)
                return;
            if (e.Delta > 0 && ClipboardTextBlock.FontSize < 80.0)
                ClipboardTextBlock.FontSize++;

            if (e.Delta < 0 && ClipboardTextBlock.FontSize > 12.0)
                ClipboardTextBlock.FontSize--;
        }

        #endregion

        private void TextBox_TextChanged(object sender, System.Windows.Controls.TextChangedEventArgs e) {
            //search phrase changed



        }

        #region MenuItemClick

        private void MIExtensions_Click(object sender, RoutedEventArgs e) {
            fileExtensionsWindow = new BaseWindow() { Owner = this };
            fileExtensionsWindow.ClientArea.Content = new FileExtensions(ref UsedExtensions, ref AvaliableExtensions, ref fileExtensionsWindow);
            fileExtensionsWindow.ShowDialog();

        }

        private void MIIgnoredInterfaces_Click(object sender, RoutedEventArgs e) {

        }

        private void MISourceInterfaces_Click(object sender, RoutedEventArgs e) {
            networkSettingsWindow = new BaseWindow() { Owner = this };
            networkSettingsWindow.ClientArea.Content = new NetworkSettings(ref UsedInterfaces, ref AvaliableInterfaces, ref networkSettingsWindow);
            networkSettingsWindow.ShowDialog();

        }

        private void NetworkSettingsWindow_modyfiedUsedInterface(InterfaceClass obj) {
            throw new NotImplementedException();
        }

        private void NetworkSettingsWindow_removedUsedInterface(InterfaceClass obj) {
            throw new NotImplementedException();
        }

        private void NetworkSettingsWindow_addedUsedInterface(InterfaceClass obj) {
            throw new NotImplementedException();
        }

        private void MINewSession_Click(object sender, RoutedEventArgs e) {

        }

        private void MISaveMultipleFiles_Click(object sender, RoutedEventArgs e) {

        }

        #endregion
    }
}
