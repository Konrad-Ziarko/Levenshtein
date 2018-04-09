using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.ComponentModel;
using System.Linq;
using System.Net.NetworkInformation;
using System.Net.Sockets;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Input;
using WPF.JoshSmith.ServiceProviders.UI;

namespace Zniffer {
    /// <summary>
    /// Interaction logic for NetworkSettings.xaml
    /// </summary>
    public partial class NetworkSettings : UserControl {

        private ObservableCollection<InterfaceClass> _UsedInterfaces;
        private ObservableCollection<InterfaceClass> _AvaliableInterfaces;
        private BaseWindow MyBaseWindow;

        public ObservableCollection<InterfaceClass> UsedInterfaces {
            get {
                return _UsedInterfaces;
            }
        }
        public ObservableCollection<InterfaceClass> AvaliableInterfaces {
            get {
                return _AvaliableInterfaces;
            }
        }

        private void button_Exit_Click(object sender, RoutedEventArgs e) {
            //Close();
        }
        public NetworkSettings(ref ObservableCollection<InterfaceClass> UsedInterfaces, ref ObservableCollection<InterfaceClass> AvaliableInterfaces, ref BaseWindow MyBaseWindow) {
            InitializeComponent();
            this._UsedInterfaces = UsedInterfaces;
            this._AvaliableInterfaces = AvaliableInterfaces;
            this.MyBaseWindow = MyBaseWindow;
            MyBaseWindow.Title = "Interfejsy sieciowe";

            MyBaseWindow.SizeChanged += MyBaseWindow_SizeChanged;


            List<string> networkInterfaces = new List<string>();

            /*
            string strIP = null;
            IPHostEntry HosyEntry = Dns.GetHostEntry((Dns.GetHostName()));
            if (HosyEntry.AddressList.Length > 0) {
                foreach (IPAddress ip in HosyEntry.AddressList) {
                    if (ip.AddressFamily == AddressFamily.InterNetwork) {
                        networkInterfaces.Add(strIP);
                        if (!this.AvaliableInterfaces.Any(x => x.Addres.Equals(strIP)) && !this.UsedInterfaces.Any(x => x.Addres.Equals(strIP)))
                            this.AvaliableInterfaces.Add(new InterfaceClass(strIP, ""));
                    }
                }
            }*/

            
            foreach (NetworkInterface adapter in NetworkInterface.GetAllNetworkInterfaces()) {
                if (adapter.NetworkInterfaceType != NetworkInterfaceType.Loopback && adapter.OperationalStatus == OperationalStatus.Up) {
                    foreach (UnicastIPAddressInformation ip in adapter.GetIPProperties().UnicastAddresses)
                        if (ip.Address.AddressFamily == AddressFamily.InterNetwork) {
                            networkInterfaces.Add(ip.Address.ToString());
                            if (!this.AvaliableInterfaces.Any(x => x.Addres.Equals(ip.Address.ToString())) && !this.UsedInterfaces.Any(x => x.Addres.Equals(ip.Address.ToString())))
                                this.AvaliableInterfaces.Add(new InterfaceClass(ip.Address.ToString(), ""));
                        }
                }
            }
            
            foreach (var interfaceObj in this.UsedInterfaces) {
                if (networkInterfaces.Contains(interfaceObj.Addres))
                    interfaceObj.InterfaceIsUp = true;
                else
                    interfaceObj.InterfaceIsUp = false;
            }
            foreach (var interfaceObj in this.AvaliableInterfaces) {
                if (networkInterfaces.Contains(interfaceObj.Addres))
                    interfaceObj.InterfaceIsUp = true;
                else
                    interfaceObj.InterfaceIsUp = false;
            }



            new ListViewDragDropManager<InterfaceClass>(LBAvaliable);
            new ListViewDragDropManager<InterfaceClass>(LBUsed);


            this.DataContext = this;
        }

        private void MyBaseWindow_SizeChanged(object sender, SizeChangedEventArgs e) {
            Height = MyBaseWindow.Height;
            Width = MyBaseWindow.Width;
        }

        private void Avaliable_MouseDoubleClick(object sender, MouseButtonEventArgs e)
        {
            if (LBAvaliable.SelectedItem != null)
            {
                //dodać do używanych
                //MessageBox.Show(LBAvaliable.SelectedItem.ToString());
                foreach (var iface in AvaliableInterfaces) {
                    if (iface.ToString().Equals(LBAvaliable.SelectedItem.ToString())) {
                        UsedInterfaces.Add(iface);
                        AvaliableInterfaces.Remove(iface);
                        break;
                    }
                }
            }

        }

        private void Used_MouseDoubleClick(object sender, MouseButtonEventArgs e) {
            //allow to edit ports and save that
            InterfaceClass li = null;
            bool removeInterfaceFromUsed = false;
            foreach (var iface in UsedInterfaces) {
                if (iface.ToString().Equals(LBUsed.SelectedItem.ToString())) {
                    li = iface;
                    if (LBUsed.SelectedItem != null) {
                        BaseWindow editInterfaceWindow = new BaseWindow() { Owner = MyBaseWindow, MaxHeight = 75, Width = 300 };
                        EditInterface editInterface = new EditInterface(ref li, ref editInterfaceWindow);
                        editInterfaceWindow.ClientArea.Content = editInterface;

                        editInterfaceWindow.Closing += new CancelEventHandler(delegate (object o, System.ComponentModel.CancelEventArgs cancelEventArgs) {
                            iface.Ports = editInterface.getPorts();
                            removeInterfaceFromUsed = !(bool)editInterface.checkBox.IsChecked;

                        });
                        editInterfaceWindow.ShowDialog();
                    }
                    break;
                }
            }
            if (removeInterfaceFromUsed && li != null ) {
                AvaliableInterfaces.Add(li);
                UsedInterfaces.Remove(li);
            }
        }

        private void Button_Left_Click(object sender, RoutedEventArgs e) {
            foreach (var iface in UsedInterfaces) {
                AvaliableInterfaces.Add(iface);
            }
            UsedInterfaces.Clear();
        }

        private void Button_Right_Click(object sender, RoutedEventArgs e) {       
            foreach (var iface in AvaliableInterfaces) {
                UsedInterfaces.Add(iface);
                
            }
            AvaliableInterfaces.Clear();
        }

        private void UserControl_KeyDown(object sender, KeyEventArgs e) {
            if (e.Key == Key.Escape)
                MyBaseWindow.Close();
        }
    }
}
