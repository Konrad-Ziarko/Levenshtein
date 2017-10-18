using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Linq;
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

namespace Zniffer
{
    /// <summary>
    /// Interaction logic for NetworkSettings.xaml
    /// </summary>
    public partial class NetworkSettings : Window
    {

        public ObservableCollection<InterfaceClass> UsedInterfaces;
        public ObservableCollection<InterfaceClass> AvaliableInterfaces;
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
            Close();
        }
        public NetworkSettings(ref ObservableCollection<InterfaceClass> UsedInterfaces, ref ObservableCollection<InterfaceClass> AvaliableInterfaces)
        {
            InitializeComponent();
            this.UsedInterfaces = UsedInterfaces;
            this.AvaliableInterfaces = AvaliableInterfaces;

            DataContext = this;
        }

        private void Avaliable_MouseDoubleClick(object sender, MouseButtonEventArgs e)
        {
            if (LBAvaliable.SelectedItem != null)
            {
                //dodać do używanych
                //MessageBox.Show(LBAvaliable.SelectedItem.ToString());
                InterfaceClass li = null;
                foreach (var iface in AvaliableFaces) {
                    if (iface.ToString().Equals(LBAvaliable.SelectedItem.ToString())) {
                        UsedFaces.Add(iface);
                        AvaliableFaces.Remove(iface);
                        break;
                    }
                }
            }

        }

        private void Used_MouseDoubleClick(object sender, MouseButtonEventArgs e)
        {

            //allow to edit ports and save that

            //((LocalInterface)LBUsed.SelectedItem).ports = "1337";

            /*
            string[] split = LBUsed.SelectedItem.ToString().Split(':');

            foreach(var item in UsedFaces)
            {
                if (item.addres.Equals(split[0]))
                {
                    item.ports = "1337";
                    break;
                }
            }*/
            if (LBUsed.SelectedItem != null) {
                BaseWindow editInterfaceWindow = new BaseWindow() { Owner = this };
                //editInterfaceWindow.Content = new EditInterface();
                editInterfaceWindow.DataContext = new EditInterface();
                editInterfaceWindow.ShowDialog();
            }



        }
    }
}
