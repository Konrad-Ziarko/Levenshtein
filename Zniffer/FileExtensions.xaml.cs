using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.ComponentModel;
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

        private ObservableCollection<FileExtensionClass> _UsedExtensions;
        private ObservableCollection<FileExtensionClass> _AvaliableExtensions;
        public ObservableCollection<FileExtensionClass> UsedExtensions {
            get {
                return _UsedExtensions;
            }
        }
        public ObservableCollection<FileExtensionClass> AvaliableExtensions {
            get {
                return _AvaliableExtensions;
            }
        }

        public FileExtensions(ref ObservableCollection<FileExtensionClass> UsedExtensions, ref ObservableCollection<FileExtensionClass> AvaliableExtensions, ref BaseWindow MyBaseWindow) {
            InitializeComponent();

            this._AvaliableExtensions = AvaliableExtensions;
            this._UsedExtensions = UsedExtensions;

            MyBaseWindow.Title = "Rozszerzenia plików";
            this.MyBaseWindow = MyBaseWindow;

            MyBaseWindow.SizeChanged += MyBaseWindow_SizeChanged;
            MyBaseWindow.Closing += MyBaseWindow_Closing;

            new ListViewDragDropManager<FileExtensionClass>(LBAvaliable);
            new ListViewDragDropManager<FileExtensionClass>(LBUsed);

            this.DataContext = this;
        }
        private void MyBaseWindow_Closing(object sender, CancelEventArgs e) {
            throw new NotImplementedException();
            //zapisac rozszerzenia do ustawien
        }

        private void MyBaseWindow_SizeChanged(object sender, SizeChangedEventArgs e) {
            Height = MyBaseWindow.ClientArea.Height;
            Width = MyBaseWindow.ClientArea.Width;
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
            foreach (FileExtensionClass obj in UsedExtensions) {
                AvaliableExtensions.Add(obj);

            }
            UsedExtensions.Clear();
        }

        private void Button_Right_Click(object sender, RoutedEventArgs e) {
            foreach (FileExtensionClass obj in AvaliableExtensions) {
                UsedExtensions.Add(obj);

            }
            AvaliableExtensions.Clear();

        }

        private void TextBox_KeyDown(object sender, KeyEventArgs e) {
            if(e.Key == Key.Enter) {
                Properties.Settings.Default.AvaliableExtensions.Add(TextBox_Extension.Text);
                
                ICollectionView view = CollectionViewSource.GetDefaultView(Properties.Settings.Default.AvaliableExtensions);
                view.Refresh();
                Properties.Settings.Default.Save();
                TextBox_Extension.Text = "";
            }
        }
    }
}
