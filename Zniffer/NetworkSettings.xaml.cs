using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.ComponentModel;
using System.Linq;
using System.Net.NetworkInformation;
using System.Net.Sockets;
using System.Text;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Data;
using System.Windows.Documents;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Shapes;
using WPF.JoshSmith.ServiceProviders.UI;

namespace Zniffer
{
    /// <summary>
    /// Interaction logic for NetworkSettings.xaml
    /// </summary>
    public partial class NetworkSettings : UserControl {

        public ObservableCollection<InterfaceClass> UsedInterfaces;
        public ObservableCollection<InterfaceClass> AvaliableInterfaces;
        private BaseWindow MyBaseWindow;

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

        private void button_Exit_Click(object sender, RoutedEventArgs e) {
            //Close();
        }
        public NetworkSettings(ref ObservableCollection<InterfaceClass> UsedInterfaces, ref ObservableCollection<InterfaceClass> AvaliableInterfaces, ref BaseWindow MyBaseWindow) {
            InitializeComponent();
            this.UsedInterfaces = UsedInterfaces;
            this.AvaliableInterfaces = AvaliableInterfaces;
            this.MyBaseWindow = MyBaseWindow;
            MyBaseWindow.Title = "Interfejsy sieciowe";

            MyBaseWindow.SizeChanged += MyBaseWindow_SizeChanged;


            List<string> networkInterfaces = new List<string>();
            foreach (NetworkInterface adapter in NetworkInterface.GetAllNetworkInterfaces()) {
                if (adapter.NetworkInterfaceType == NetworkInterfaceType.Ethernet && adapter.OperationalStatus == OperationalStatus.Up) {
                    foreach (UnicastIPAddressInformation ip in adapter.GetIPProperties().UnicastAddresses)
                        if (ip.Address.AddressFamily == AddressFamily.InterNetwork) {
                            networkInterfaces.Add(ip.Address.ToString());
                            if (!AvaliableFaces.Any(x => x.Addres.Equals(ip.Address.ToString())) && !UsedFaces.Any(x => x.Addres.Equals(ip.Address.ToString())))
                                AvaliableFaces.Add(new InterfaceClass(ip.Address.ToString(), ""));
                        }
                }
            }
            foreach (var interfaceObj in UsedFaces) {
                if (networkInterfaces.Contains(interfaceObj.Addres))
                    interfaceObj.InterfaceIsUp = true;
                else
                    interfaceObj.InterfaceIsUp = false;
            }
            foreach (var interfaceObj in AvaliableFaces) {
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
                foreach (var iface in AvaliableFaces) {
                    if (iface.ToString().Equals(LBAvaliable.SelectedItem.ToString())) {
                        UsedFaces.Add(iface);
                        AvaliableFaces.Remove(iface);
                        break;
                    }
                }
            }

        }

        private void Used_MouseDoubleClick(object sender, MouseButtonEventArgs e) {
            //allow to edit ports and save that
            InterfaceClass li = null;
            bool removeInterfaceFromUsed = false;
            foreach (var iface in UsedFaces) {
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
                AvaliableFaces.Add(li);
                UsedFaces.Remove(li);
            }
        }

        private void Button_Left_Click(object sender, RoutedEventArgs e) {
            foreach (var iface in UsedFaces) {
                AvaliableFaces.Add(iface);
            }
            UsedFaces.Clear();
        }

        private void Button_Right_Click(object sender, RoutedEventArgs e) {       
            foreach (var iface in AvaliableFaces) {
                UsedFaces.Add(iface);
                
            }
            AvaliableFaces.Clear();
        }
    }
}
