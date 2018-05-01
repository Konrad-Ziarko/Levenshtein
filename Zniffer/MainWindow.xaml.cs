/// Written by Konrad Ziarko
/// konrad.tomasz.ziarko@gmail.com
/// 
/// Feel free to copy anything you find :)
/// 
/// Sniffer code from CodeProject.com article "SharpPcap - A Packet Capture Framework for .NET"
/// link:https://www.codeproject.com/Articles/12458/SharpPcap-A-Packet-Capture-Framework-for-NET
/// 
/// Follow me on GitHub @Konrad-Ziarko
/// 


using CustomExtensions;
using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.IO;
using System.Linq;
using System.Management;
using System.Net.NetworkInformation;
using System.Net.Sockets;
using System.Runtime.InteropServices;
using System.Text;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Documents;
using System.Windows.Input;
using System.Windows.Interop;
using System.Windows.Media;
using Zniffer.FileExtension;
using Zniffer.FilesAndText;
using Zniffer.Levenshtein;

namespace Zniffer {

    /// <summary>
    /// Interaction logic for MainWindow.xaml
    /// </summary>
    public partial class MainWindow : Window {
        //public static LevenshteinMode SearchMode = LevenshteinMode.SplitDualRowCPU;

        private static bool isScanerInitialized = false;
        private static bool isSnifferInitialized = false;
        private static bool isKeyLoggerInitialized = false;

        private static bool AutoScrollClipboard = true;
        private ManagementEventWatcher watcher = new ManagementEventWatcher();

        #region Levenshtein Prop
        public LevenshteinMode snifferMode {
            get {
                return _snifferMode;
            }
            set {
                _snifferMode = value;
            }
        }
        public LevenshteinMode keyloggerMode {
            get {
                return _keyloggerMode;
            }
            set {
                _scanerMode = value;
            }
        }
        public LevenshteinMode scanerMode {
            get {
                return _scanerMode;
            }
            set {
                _scanerMode = value;
            }
        }
        private LevenshteinMode _snifferMode;
        private LevenshteinMode _keyloggerMode;
        private LevenshteinMode _scanerMode;
        #endregion

        #region Networking

        public static Dictionary<string, string> AvaliableNetworkAdapters = new Dictionary<string, string>();
        public ObservableCollection<InterfaceClass> UsedInterfaces = new ObservableCollection<InterfaceClass>();
        public ObservableCollection<InterfaceClass> AvaliableInterfaces = new ObservableCollection<InterfaceClass>();


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
            sniffer.removeAllConnections();
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

        private IntPtr WndProc(IntPtr hwnd, int msg, IntPtr wParam, IntPtr lParam, ref bool handled) {
            if (msg == 0x0308) {
                //wyciąganie danych z obrazów i dźwięków
                IDataObject iData = new DataObject();

                try {
                    iData = Clipboard.GetDataObject();
                }
                catch (ExternalException externEx) {
                    Console.Out.WriteLine("InteropServices.ExternalException: {0}", externEx.Message);

                    return IntPtr.Zero; ;
                }
                catch (Exception) {
                    return IntPtr.Zero; ;
                }
                if (iData.GetDataPresent(DataFormats.Text) || iData.GetDataPresent(DataFormats.Rtf) || iData.GetDataPresent(DataFormats.UnicodeText)) {
                    try {
                        string _data = (string)iData.GetData(DataFormats.Text);
                        Console.Out.WriteLine((string)iData.GetData(DataFormats.Text));
                        string phrase = SearchPhrase;
                        //way to change levenshtein methode
                        LevenshteinMatches results = _data.Levenshtein(phrase, mode: keyloggerMode);

                        if (results != null && results.hasMatches) {
                            AddTextToClipBoardBox(results);
                        }
                    }
                    catch (System.Runtime.InteropServices.COMException) { }

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

            //run sniffer
            sniffer = new Sniffer(this, ref UsedInterfaces);
            //run usb scanner
            searcher = new Searcher(this);
            //run keylogger
            keyLogger = new KeyLogger(this);
        }


        //discovering new drives
        private void NewDeviceDetectedEventArived(object sender, EventArrivedEventArgs e) {

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
            if (new WindowInteropHelper(this).Handle != IntPtr.Zero) {
                clipboardViewerNext = SetClipboardViewer(new WindowInteropHelper(this).Handle);
                HwndSource source = HwndSource.FromHwnd(new WindowInteropHelper(this).Handle);
                source.AddHook(new HwndSourceHook(WndProc));
            }
        }

        public void detachFromClipboard() {
            ChangeClipboardChain(new WindowInteropHelper(this).Handle, clipboardViewerNext);
        }


        public void Window_SourceInitialized(object sender, EventArgs e) {

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
                catch (IOException) {
                    //removed disconnected drives from list
                }
                catch (ArgumentException) {
                    //connected drive with same letter;
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
            }

            //attach clipboard listener
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

            sniffer.endQueueThread();
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
                string[] parts = new string[] { match.contextL, match.value, match.contextR };
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
                runs.Add(new Run("「" + match.distance + "」\n"));
                foreach (var item in runs)
                    NetworkTextBlock.Inlines.Add(item);
            }
            NetworkTextBlock.Inlines.Add(new Run("\n"));
        }

        public void AddTextToNetworkBox(string txt) {
            if (txt != null && txt.Length != 0) {
                if (!NetworkTextBlock.Dispatcher.CheckAccess()) {
                    NetworkTextBlock.Dispatcher.Invoke(() => {
                        NetworkTextBlock.Inlines.Add(new Run(txt) { Foreground = new SolidColorBrush(Color.FromRgb(10, 170, 20)) });
                    });
                }
                else {
                    NetworkTextBlock.Inlines.Add(new Run(txt) { Foreground = new SolidColorBrush(Color.FromRgb(10, 170, 20)) });
                }
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
                string[] parts = new string[] { match.contextL, match.value, match.contextR };
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
                runs.Add(new Run("「" + match.distance + "」\n"));
                foreach (var item in runs)
                    ClipboardTextBlock.Inlines.Add(item);
            }
            ClipboardTextBlock.Inlines.Add(new Run("\n"));
        }
        public void AddTextToClipBoardBox(LevenshteinMatches matches) {
            if (matches.hasMatches) {
                if (!ClipboardTextBlock.Dispatcher.CheckAccess()) {
                    try {
                        ClipboardTextBlock.Dispatcher.Invoke(() => {
                            _AddTextToClipBoardBox(matches);
                        });
                    }
                    catch { }
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
                    });
                }
                else {
                    FilesTextBlock.Inlines.Add(new Run(txt));
                }
            }
        }

        private void _AddTextToFileBox(LevenshteinMatches matches) {
            List<Run> runs = new List<Run>();
            foreach (LevenshteinMatch match in matches.foundMatches) {
                string[] parts = new string[] { match.contextL, match.value, match.contextR };
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
                runs.Add(new Run("「" + match.distance + "」\n"));
                foreach (var item in runs)
                    FilesTextBlock.Inlines.Add(item);
            }
            FilesTextBlock.Inlines.Add(new Run("\n"));
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
            networkSettingsWindow.ClientArea.Content = new NetworkSettings(ref UsedInterfaces, ref AvaliableInterfaces, ref networkSettingsWindow);
            //stop sniffer
            if (MISniff.IsChecked) {
                MISniff.IsChecked = false;
                MISniff_Click(this, e);
            }

            networkSettingsWindow.ShowDialog();
        }

        private void NetworkSettingsWindow_Closing(object sender, System.ComponentModel.CancelEventArgs e) {
        }

        private void MINewSession_Click(object sender, RoutedEventArgs e) {
            NetworkTextBlock.Inlines.Clear();
            ClipboardTextBlock.Inlines.Clear();
            FilesTextBlock.Inlines.Clear();
        }

        private void NetworkScrollViewr_ScrollChanged(object sender, System.Windows.Controls.ScrollChangedEventArgs e) {

        }

        private void MISniff_Click(object sender, RoutedEventArgs e) {
            //turn on/off sniffer
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


        private void ZnifferWindow_Loaded(object sender, RoutedEventArgs e) {

            #region experiment

            var csv = new StringBuilder();

            string lorem = "Lorem ipsum dolor sit amet, consectetur adipisicin";//len = 50
            int len = lorem.Length;
            int multi = 80;

            string expression = "zniffer";
            //int strLen = str.Length;
            int exprLen = expression.Length;

            csv.AppendLine("len,Macierz trójwymiarowa na GPU,Macierz trójwymiarowa na CPU,Dzielenie ciągu na wyrazy na CPU,Dzielenie ciągu na wyrazy macierz dwu wierszowa na CPU,Macierz trójwymiarowa na CPU z wieloma wątkami,Dzielenie ciągu na wyrazy na CPU z wieloma wątkami,Histogram poprzedzający macierz na CPU");

            string str = string.Concat(Enumerable.Repeat(lorem, multi));
            str += "sniffer";

            int loops = 1;

            //81920
            //163840

            for (; multi <= 40960; multi *= 2) {

                var watch = System.Diagnostics.Stopwatch.StartNew();

                csv.Append(str.Length + ",");

                watch.Reset();
                for (int i = 0; i < loops; i++) {
                    watch.Start();
                    var wynik3 = str.Levenshtein(expression, mode: LevenshteinMode.ThreeDimMatrixGPU);
                    watch.Stop();
                }
                Console.WriteLine("gpu " + str.Length + " " + watch.ElapsedMilliseconds);
                //csv.Append(watch.ElapsedMilliseconds + ",");

                //watch.Reset();
                //for (int i = 0; i < loops; i++) {
                //    watch.Start();
                //    var wynik2 = str.Levenshtein(expression, mode: LevenshteinMode.ThreeDimMatrixCPU);
                //    watch.Stop();
                //}
                //Console.WriteLine(str.Length + " " + watch.ElapsedMilliseconds);
                //csv.Append(watch.ElapsedMilliseconds + ",");

                //watch.Reset();
                //for (int i = 0; i < loops; i++) {
                //    watch.Start();
                //    var wynik2 = str.Levenshtein(expression, mode: LevenshteinMode.SplitForSingleMatrixCPU);
                //    watch.Stop();
                //}
                //Console.WriteLine(str.Length + " " + watch.ElapsedMilliseconds);
                //csv.Append(watch.ElapsedMilliseconds + ",");

                //watch.Reset();
                //for (int i = 0; i < loops; i++) {
                //    watch.Start();
                //    var wynik2 = str.Levenshtein(expression, mode: LevenshteinMode.SplitDualRowCPU);
                //    watch.Stop();
                //}
                //Console.WriteLine(str.Length + " " + watch.ElapsedMilliseconds);
                //csv.Append(watch.ElapsedMilliseconds + ",");

                //watch.Reset();
                //for (int i = 0; i < loops; i++) {
                //    watch.Start();
                //    var wynik2 = str.Levenshtein(expression, mode: LevenshteinMode.ThreeDimMatrixParallelCPU);
                //    watch.Stop();
                //}
                //Console.WriteLine(str.Length + " " + watch.ElapsedMilliseconds);
                //csv.Append(watch.ElapsedMilliseconds + ",");

                //watch.Reset();
                //for (int i = 0; i < loops; i++) {
                //    watch.Start();
                //    var wynik2 = str.Levenshtein(expression, mode: LevenshteinMode.SplitForParallelCPU);
                //    watch.Stop();
                //}
                //Console.WriteLine(str.Length + " " + watch.ElapsedMilliseconds);
                //csv.Append(watch.ElapsedMilliseconds + ",");


                //watch.Reset();
                //for (int i = 0; i < loops; i++) {
                //    watch.Start();
                //    var wynik2 = str.Levenshtein(expression, mode: LevenshteinMode.HistogramCPU);
                //    watch.Stop();
                //}
                //Console.WriteLine("hist " + str.Length + " " + watch.ElapsedMilliseconds);
                //csv.Append(watch.ElapsedMilliseconds + "");

                csv.AppendLine("");

                //File.AppendAllText(@"C:\Users\Konrad\Downloads\wyniki5.csv", csv.ToString());
                //csv.Clear();

                GC.Collect();

                str = string.Concat(Enumerable.Repeat(str, 2));
            }


            #endregion
        }

        private void MIAlgo_Click(object sender, RoutedEventArgs e) {

        }

        private void NetworkScrollViewr_MouseRightButtonDown(object sender, MouseButtonEventArgs e) {
            ContextMenu menu = new ContextMenu();
            menu.Items.Add("asd");
            menu.Items.Add("asd");
            menu.Visibility = Visibility.Visible;
        }
        public static IEnumerable<T> FindVisualChildren<T>(DependencyObject depObj) where T : DependencyObject {
            if (depObj != null) {
                for (int i = 0; i < VisualTreeHelper.GetChildrenCount(depObj); i++) {
                    DependencyObject child = VisualTreeHelper.GetChild(depObj, i);
                    if (child != null && child is T) {
                        yield return (T)child;
                    }

                    foreach (T childOfChild in FindVisualChildren<T>(child)) {
                        yield return childOfChild;
                    }
                }
            }
        }
        //network
        private void ContextMenu_Click(object sender, RoutedEventArgs e) {
            ContextMenu sen = sender as ContextMenu;
            var x = sen.Items.SourceCollection;
            MenuItem mi = e.OriginalSource as MenuItem;
            foreach (MenuItem item in FindVisualChildren<MenuItem>(sen)) {
                item.IsChecked = false;
            }
            mi.IsChecked = true;
            string context = mi.DataContext.ToString();

            Enum.TryParse(context, out _snifferMode);
        }
        private void ContextMenu_Loaded(object sender, RoutedEventArgs e) {
            if (!isKeyLoggerInitialized) {
                ContextMenu sen = sender as ContextMenu;
                var item = FindVisualChildren<MenuItem>(sen).First();
                item.IsChecked = isKeyLoggerInitialized = true;
            }
        }
        //
        //keylogger
        private void ContextMenu_Click_1(object sender, RoutedEventArgs e) {
            ContextMenu sen = sender as ContextMenu;
            var x = sen.Items.SourceCollection;
            MenuItem mi = e.OriginalSource as MenuItem;
            foreach (MenuItem item in FindVisualChildren<MenuItem>(sen)) {
                item.IsChecked = false;
            }
            mi.IsChecked = true;
            string context = mi.DataContext.ToString();

            Enum.TryParse(context, out _keyloggerMode);
        }
        private void ContextMenu_Loaded_1(object sender, RoutedEventArgs e) {
            if (!isSnifferInitialized) {
                ContextMenu sen = sender as ContextMenu;
                var item = FindVisualChildren<MenuItem>(sen).First();
                item.IsChecked = isSnifferInitialized = true;
            }
        }
        //
        //scaner
        private void ContextMenu_Click_2(object sender, RoutedEventArgs e) {
            ContextMenu sen = sender as ContextMenu;
            var x = sen.Items.SourceCollection;
            MenuItem mi = e.OriginalSource as MenuItem;
            foreach (MenuItem item in FindVisualChildren<MenuItem>(sen)) {
                item.IsChecked = false;
            }
            mi.IsChecked = true;
            string context = mi.DataContext.ToString();

            Enum.TryParse(context, out _scanerMode);
        }
        private void ContextMenu_Loaded_2(object sender, RoutedEventArgs e) {
            if (!isScanerInitialized) {
                ContextMenu sen = sender as ContextMenu;
                var item = FindVisualChildren<MenuItem>(sen).First();
                item.IsChecked = isScanerInitialized = true;
            }
        }
        //

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
