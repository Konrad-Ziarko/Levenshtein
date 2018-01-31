using CustomExtensions;
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
using System.Windows.Documents;
using System.Windows.Input;
using System.Windows.Interop;
using System.Windows.Media;
using Trinet.Core.IO.Ntfs;
using Zniffer.FileExtension;
using Zniffer.FilesAndText;
using Zniffer.Levenshtein;

namespace Zniffer {

    /// <summary>
    /// Interaction logic for MainWindow.xaml
    /// </summary>
    public partial class MainWindow : Window {
        private static MainWindow THISREF = null;
        public static string COLORTAG = "!@#RED$%^";

        private static bool AutoScrollClipboard = true;
        private ManagementEventWatcher watcher = new ManagementEventWatcher();

        #region Networking
        private static bool _TCP;
        private static bool _UDP;
        public static bool TCP {
            get {
                return _TCP;
            }
            set {
                _TCP = value;
            }
        }
        public static bool UDP {
            get {
                return _UDP;
            }
            set {
                _UDP = value;
            }
        }
        private static bool _SaveFile;
        public static bool SaveFile {
            get {
                return _SaveFile;
            }
            set {
                _SaveFile = value;
            }
        }


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
        private void Button_Exit_Click(object sender, RoutedEventArgs e) {
            Close();
        }

        private void Button_Max_Click(object sender, RoutedEventArgs e) {
            if (WindowState == WindowState.Maximized)
                WindowState = WindowState.Normal;
            else
                WindowState = WindowState.Maximized;
        }

        private void Button_Min_Click(object sender, RoutedEventArgs e) {
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
        public void TempPisanie() {
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
        public void TempCzytanie() {
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

        private KeyLogger keyLogger;
        private Sniffer sniffer;
        private Searcher searcher;

        public Dictionary<string, string> avaliableDrives = new Dictionary<string, string>();

        private static string _SearchPhrase = "Zniffer";
        public static string SearchPhrase {
            get {
                return _SearchPhrase;
            }
            set {
                _SearchPhrase = value;
            }
        }



        public MainWindow() {
            THISREF = this;
            InitializeComponent();
            this.DataContext = this;

            //initialize settings collections if needed
            if (Properties.Settings.Default.UsedExtensions == null)
                Properties.Settings.Default.UsedExtensions = new System.Collections.Specialized.StringCollection();
            if (Properties.Settings.Default.AvaliableExtensions == null)
                Properties.Settings.Default.AvaliableExtensions = new System.Collections.Specialized.StringCollection();

            //load collections from settings
            foreach (string ext in Properties.Settings.Default.AvaliableExtensions)
                AvaliableExt.Add(new FileExtensionClass(ext));
            foreach (string ext in Properties.Settings.Default.UsedExtensions)
                UsedExt.Add(new FileExtensionClass(ext));

            //TODO implement sniffer
            sniffer = new Sniffer(this, ref UsedInterfaces);
            searcher = new Searcher(this);
            //run keylogger
            keyLogger = new KeyLogger(this);
        }


        //discovering new drives
        private void NewDeviceDetectedEventArived(object sender, EventArrivedEventArgs e) {

            //TODO async scan files on newly attached devices (if ntfs +ADS)

            Console.Out.WriteLine(e.NewEvent.Properties["DriveName"].Value.ToString() + " inserted");

            foreach (DriveInfo drive in DriveInfo.GetDrives()) {
                if (drive.Name.Equals(e.NewEvent.Properties["DriveName"].Value.ToString() + "\\")) {
                    List<string> directories = searcher.GetDirectories(drive.Name);

                    List<string> files = searcher.GetFiles(drive.Name);
                    searcher.SearchFiles(files, drive);

                    foreach (string directory in directories) {
                        //Console.Out.WriteLine(directory);

                        files = searcher.GetFiles(directory);
                        searcher.SearchFiles(files, drive);
                    }
                }

            }
        }

        public void attachToClipboard() {
            //attach to clipboard
            if((IntPtr)(new WindowInteropHelper(this).Handle) != IntPtr.Zero) {
                clipboardViewerNext = SetClipboardViewer(new WindowInteropHelper(this).Handle);
                HwndSource source = HwndSource.FromHwnd(new WindowInteropHelper(this).Handle);
                source.AddHook(new HwndSourceHook(WndProc));
            }
        }

        public void detachFromClipboard() {
            ChangeClipboardChain(new WindowInteropHelper(this).Handle, clipboardViewerNext);
        }


        public void Window_SourceInitialized(object sender, EventArgs e) {
            ////
            /*
            string s = "abcabc def ghi jkl l";
            //s = string.Concat(Enumerable.Repeat(s, 5000));

            var expression = "ghj";

            var watch = System.Diagnostics.Stopwatch.StartNew();
            //long memory = GC.GetTotalMemory(true);
            var res = s.Levenshtein(expression, mode: LevenshteinMode.SplitForSingleMatrixCPU);
            //long memory2 = GC.GetTotalMemory(true);
            watch.Stop();

            var elapsedMs = watch.ElapsedMilliseconds;

            var watch2 = System.Diagnostics.Stopwatch.StartNew();
            //long memory = GC.GetTotalMemory(true);
            var res2 = s.Levenshtein(expression, mode: LevenshteinMode.SplitForSingleMatrixCPU);
            //long memory2 = GC.GetTotalMemory(true);
            watch2.Stop();

            var elapsedMs2 = watch2.ElapsedMilliseconds;
            */
            ////



            //check file system // if ntfs look for ADSs
            foreach (DriveInfo d in DriveInfo.GetDrives()) {
                //Console.Out.WriteLine("Drive {0}", d.Name);
                //Console.Out.WriteLine("  Drive type: {0}", d.DriveType.ToString());
                if (d.IsReady == true) {
                    //Console.Out.WriteLine("  Volume label: {0}", d.VolumeLabel);
                    //Console.Out.WriteLine("  File system: {0}", d.DriveFormat);
                }
                try {
                    avaliableDrives.Add(d.Name, d.DriveFormat);
                }
                catch (ArgumentException) {
                    //connected drive with same letter;
                    //remove disconnected drives from list
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
                //Console.Out.WriteLine(description.Value.ToString());
            }

            //check for the amount of ram
            double amountOfRam = new Microsoft.VisualBasic.Devices.ComputerInfo().TotalPhysicalMemory;
            double amountOfRamInMB = amountOfRam / 1024 / 1024;


            //detect new network connections/interfaces
            NetworkChange.NetworkAddressChanged += new NetworkAddressChangedEventHandler(AddressChangedCallback);

            //look for network adapters
            NetworkInterface[] adapters = NetworkInterface.GetAllNetworkInterfaces();
            foreach (NetworkInterface adapter in adapters) {
                string ipAddrList = string.Empty;
                IPInterfaceProperties properties = adapter.GetIPProperties();
                if (adapter.NetworkInterfaceType == NetworkInterfaceType.Ethernet && adapter.OperationalStatus == OperationalStatus.Up) {
                    foreach (UnicastIPAddressInformation ip in adapter.GetIPProperties().UnicastAddresses)
                        if (ip.Address.AddressFamily == AddressFamily.InterNetwork) {
                            AvaliableInterfaces.Add(new InterfaceClass(ip.Address.ToString(), ""));
                        }
                }
                //Console.Out.WriteLine("\n");
            }


            //attach to clipboard
            if (MIClipBoard.IsChecked)
                attachToClipboard();
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
            detachFromClipboard();

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
            SearchPhrase = SearchPhraseTextBox.Text;
        }

        #region AddTextTo
        private void _AddTextToNetworkBox(LevenshteinMatches matches) {
            List<Run> runs = new List<Run>();
            foreach (LevenshteinMatch match in matches.foundMatches) {
                string[] parts = match.context.Split(new string[] { "<" + COLORTAG + ">", "</" + COLORTAG + ">" }, StringSplitOptions.None);
                int i = 0;
                foreach (string s in parts) {
                    i = (i + 1) % 2;
                    if (i == 0)
                        runs.Add(new Run(s) {
                            Foreground = new SolidColorBrush(Color.FromRgb(
                                (byte)(match.length - match.distance).Map(0, match.length, 100, 255),
                                (byte)(match.length - match.distance).Map(0, match.length, 100, 255),
                                0))
                        });
                    else
                        runs.Add(new Run(s));
                }
                runs.Add(new Run("\r\n"));
                foreach (var item in runs)
                    NetworkTextBlock.Inlines.Add(item);
                NetworkTextBlock.Inlines.Add(new Run("\r\n"));
            }
        }
        public void AddTextToNetworkBox(LevenshteinMatches matches) {
            if (matches.hasMatches) {
                if (!NetworkTextBlock.Dispatcher.CheckAccess()) {
                    NetworkTextBlock.Dispatcher.Invoke(() => {
                        _AddTextToNetworkBox(matches);
                    });
                }
                else {
                    _AddTextToNetworkBox(matches);
                }
            }
        }

        private void _AddTextToClipBoardBox(LevenshteinMatches matches) {
            List<Run> runs = new List<Run>();
            foreach (LevenshteinMatch match in matches.foundMatches) {
                string[] parts = match.context.Split(new string[] { "<" + COLORTAG + ">", "</" + COLORTAG + ">" }, StringSplitOptions.None);
                int i = 0;
                foreach (string s in parts) {
                    i = (i + 1) % 2;
                    if (i == 0)
                        runs.Add(new Run(s) {
                            Foreground = new SolidColorBrush(Color.FromRgb(
                                (byte)(match.length - match.distance).Map(0, match.length, 100, 255), 
                                (byte)(match.length - match.distance).Map(0, match.length, 100, 255), 
                                0))
                        });
                    else
                        runs.Add(new Run(s));
                }
                runs.Add(new Run("「" + match.distance + "」\r\n"));
                foreach (var item in runs)
                    ClipboardTextBlock.Inlines.Add(item);
                ClipboardTextBlock.Inlines.Add(new Run("\r\n"));
            }
        }
        public void AddTextToClipBoardBox(LevenshteinMatches matches) {
            if (matches.hasMatches) {
                if (!ClipboardTextBlock.Dispatcher.CheckAccess()) {
                    ClipboardTextBlock.Dispatcher.Invoke(() => {
                        _AddTextToClipBoardBox(matches);
                    });
                }
                else {
                    _AddTextToClipBoardBox(matches);
                }
            }
        }
        public void AddTextToFileBox(string txt) {
            if (txt != null && txt.Length != 0) {
                if (!FilesTextBlock.Dispatcher.CheckAccess()) {
                    FilesTextBlock.Dispatcher.Invoke(() => {
                        FilesTextBlock.Inlines.Add(new Run(txt));
                        FilesTextBlock.Inlines.Add(new Run("\r\n"));
                    });
                }
                else {
                    FilesTextBlock.Inlines.Add(new Run(txt));
                    FilesTextBlock.Inlines.Add(new Run("\r\n"));
                }
            }
        }

        private void _AddTextToFileBox(LevenshteinMatches matches) {
            List<Run> runs = new List<Run>();
            foreach (LevenshteinMatch match in matches.foundMatches) {
                string[] parts = match.context.Split(new string[] { "<" + COLORTAG + ">", "</" + COLORTAG + ">" }, StringSplitOptions.None);
                int i = 0;
                foreach (string s in parts) {
                    i = (i + 1) % 2;
                    if (i == 0)
                        runs.Add(new Run(s) { Foreground = new SolidColorBrush(Color.FromArgb((byte)(match.length - match.distance).Map(0, match.length, 50, 255), (byte)(match.length - match.distance).Map(0, match.length, 50, 255), 0, 0)) });
                    else
                        runs.Add(new Run(s));
                }
                runs.Add(new Run("\r\n"));
                foreach (var item in runs)
                    FilesTextBlock.Inlines.Add(item);
                FilesTextBlock.Inlines.Add(new Run("\r\n"));
            }
        }
        public void AddTextToFileBox(LevenshteinMatches matches) {
            if (matches.hasMatches) {
                if (!FilesTextBlock.Dispatcher.CheckAccess()) {
                    FilesTextBlock.Dispatcher.Invoke(() => {
                        _AddTextToFileBox(matches);
                    });
                }
                else {
                    _AddTextToFileBox(matches);
                }
            }
        }
        #endregion

        #region MenuItemClick

        private void MIExtensions_Click(object sender, RoutedEventArgs e) {
            fileExtensionsWindow = new BaseWindow() { Owner = this };
            fileExtensionsWindow.ClientArea.Content = new FileExtensions(ref UsedExtensions, ref AvaliableExtensions, ref fileExtensionsWindow);
            fileExtensionsWindow.ShowDialog();

        }

        private void MISourceInterfaces_Click(object sender, RoutedEventArgs e) {
            networkSettingsWindow = new BaseWindow() { Owner = this };
            networkSettingsWindow.Closing += NetworkSettingsWindow_Closing;

            networkSettingsWindow.ClientArea.Content = new NetworkSettings(ref UsedInterfaces, ref AvaliableInterfaces, ref networkSettingsWindow);
            networkSettingsWindow.ShowDialog();

            ////state changne stop listening
        }

        private void NetworkSettingsWindow_Closing(object sender, System.ComponentModel.CancelEventArgs e) {
            //compare avaliable and used interfaces
            //state changne stop listening
            //throw new NotImplementedException();

            //InterfaceClass tmp = UsedInterfaces[0];

            //sniffer.newInterfaceAdded(tmp);
        }

        private void MINewSession_Click(object sender, RoutedEventArgs e) {

        }

        private void NetworkScrollViewr_ScrollChanged(object sender, System.Windows.Controls.ScrollChangedEventArgs e) {

        }

        private void MISniff_Click(object sender, RoutedEventArgs e) {
            if (MISniff.IsChecked) {
                sniffer.addAllConnections();
            }
            else {
                sniffer.removeAllConnections();
            }
        }

        private void MITCP_Click(object sender, RoutedEventArgs e) {
            //state changne stop listening
            if (MISniff.IsChecked) {
                MISniff.IsChecked = false;
                MISniff_Click(sender, e);
            }
        }

        private void MIUDP_Click(object sender, RoutedEventArgs e) {
            //state changne stop listening
            if (MISniff.IsChecked) {
                MISniff.IsChecked = false;
                MISniff_Click(sender, e);
            }
        }

        private void MIClipBoard_Click(object sender, RoutedEventArgs e) {
            Console.Out.WriteLine("Schowek " + MIClipBoard.IsChecked);
        }

        private void MIClipBoard_Unchecked(object sender, RoutedEventArgs e) {
            detachFromClipboard();
        }

        private void MIClipBoard_Checked(object sender, RoutedEventArgs e) {
            attachToClipboard();
        }

        private void MIDrives_Checked(object sender, RoutedEventArgs e) {
            //detect flash memory
            WqlEventQuery query = new WqlEventQuery("SELECT * FROM Win32_VolumeChangeEvent WHERE EventType = 2");
            watcher.EventArrived += new EventArrivedEventHandler(NewDeviceDetectedEventArived);
            watcher.Query = query;
            watcher.Start();
        }

        private void MIDrives_Unchecked(object sender, RoutedEventArgs e) {
            watcher.Stop();
        }

        private void MISaveFile_Click(object sender, RoutedEventArgs e) {
            if (MISniff.IsChecked) {
                MISniff.IsChecked = false;
                MISniff_Click(sender, e);
            }
            else {

            }

        }

        #endregion


        private void ClipboardScrollViewer_ScrollChanged(object sender, System.Windows.Controls.ScrollChangedEventArgs e) {
            if (e.ExtentHeightChange == 0) {
                if (ClipboardScrollViewer.VerticalOffset == ClipboardScrollViewer.ScrollableHeight) {
                    AutoScrollClipboard = true;
                }
                else {
                    AutoScrollClipboard = false;
                }
            }
            if (AutoScrollClipboard && e.ExtentHeightChange != 0) {
                ClipboardScrollViewer.ScrollToVerticalOffset(ClipboardScrollViewer.ExtentHeight);
            }
        }
    }
}
