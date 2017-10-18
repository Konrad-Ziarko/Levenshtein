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
using WPF.JoshSmith.ServiceProviders.UI;

namespace Zniffer {
    /// <summary>
    /// Interaction logic for FileExtensions.xaml
    /// </summary>
    public partial class FileExtensions : UserControl {

        private BaseWindow MyBaseWindow;

        public FileExtensions(ref BaseWindow MyBaseWindow) {
            InitializeComponent();
            MyBaseWindow.Title = "Rozszerzenia plików";
            this.MyBaseWindow = MyBaseWindow;

            MyBaseWindow.SizeChanged += MyBaseWindow_SizeChanged;

            new ListViewDragDropManager<InterfaceClass>(LBAvaliable);
            new ListViewDragDropManager<InterfaceClass>(LBUsed);


            this.DataContext = this;
        }

        private void MyBaseWindow_SizeChanged(object sender, SizeChangedEventArgs e) {
            Height = MyBaseWindow.Height;
            Width = MyBaseWindow.Width;
        }


        private void Avaliable_MouseDoubleClick(object sender, MouseButtonEventArgs e) {
            if (LBAvaliable.SelectedItem != null) {
                
            }
        }

        private void Used_MouseDoubleClick(object sender, MouseButtonEventArgs e) {
            if (LBUsed.SelectedItem != null) {

            }
        }

        private void Button_Left_Click(object sender, RoutedEventArgs e) {
            foreach(string s in Properties.Settings.Default.UsedExtensions)
                Properties.Settings.Default.AvaliableExtensions.Add(s);
            Properties.Settings.Default.UsedExtensions.Clear();
        }

        private void Button_Right_Click(object sender, RoutedEventArgs e) {
            foreach (string s in Properties.Settings.Default.UsedExtensions)
                Properties.Settings.Default.AvaliableExtensions.Add(s);
            Properties.Settings.Default.AvaliableExtensions.Clear();

        }
    }
}
