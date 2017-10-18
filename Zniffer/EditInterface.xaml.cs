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

        public EditInterface(ref BaseWindow MyBaseWindow) {
            InitializeComponent();
            this.MyBaseWindow = MyBaseWindow;

            MyBaseWindow.SizeChanged += MyBaseWindow_SizeChanged;
        }

        private void MyBaseWindow_SizeChanged(object sender, SizeChangedEventArgs e) {
            Height = MyBaseWindow.Height;
            Width = MyBaseWindow.Width;
        }
    }
}
