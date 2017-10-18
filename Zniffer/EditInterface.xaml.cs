using System;
using System.Collections.Generic;
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
using System.Windows.Navigation;
using System.Windows.Shapes;

namespace Zniffer {
    /// <summary>
    /// Interaction logic for EditInterface.xaml
    /// </summary>
    public partial class EditInterface : UserControl {
        private BaseWindow MyBaseWindow;
        public InterfaceClass interfaceObj;

        public EditInterface(ref InterfaceClass interfaceObj, ref BaseWindow MyBaseWindow) {
            InitializeComponent();
            this.MyBaseWindow = MyBaseWindow;
            MyBaseWindow.Title = "Edycja portów";

            MyBaseWindow.SizeChanged += MyBaseWindow_SizeChanged;

            this.interfaceObj = interfaceObj;
            strIP.Text = interfaceObj.addres;
            strPort.Text = interfaceObj.ports;
        }

        private void MyBaseWindow_SizeChanged(object sender, SizeChangedEventArgs e) {
            Height = MyBaseWindow.Height;
            Width = MyBaseWindow.Width;
        }

        public string getPorts() {
            return strPort.Text;
        }
    }
}
