using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.ComponentModel;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Input;
using WPF.JoshSmith.ServiceProviders.UI;
using Zniffer.FileExtension;

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

            Properties.Settings.Default.AvaliableExtensions = new System.Collections.Specialized.StringCollection();
            Properties.Settings.Default.UsedExtensions = new System.Collections.Specialized.StringCollection();

            foreach (var obj in AvaliableExtensions)
                Properties.Settings.Default.AvaliableExtensions.Add(obj.Extension);
            foreach (var obj in UsedExtensions)
                Properties.Settings.Default.UsedExtensions.Add(obj.Extension);

            Properties.Settings.Default.Save();
        }

        private void MyBaseWindow_SizeChanged(object sender, SizeChangedEventArgs e) {
            Height = MyBaseWindow.ClientArea.Height;
            Width = MyBaseWindow.ClientArea.Width;
        }


        private void Avaliable_MouseDoubleClick(object sender, MouseButtonEventArgs e) {
            if (LBAvaliable.SelectedItem != null) {
                foreach (var ext in AvaliableExtensions) {
                    if (ext.ToString().Equals(LBAvaliable.SelectedItem.ToString())) {
                        UsedExtensions.Add(ext);
                        AvaliableExtensions.Remove(ext);
                        break;
                    }
                }
            }
        }

        private void Used_MouseDoubleClick(object sender, MouseButtonEventArgs e) {
            if (LBUsed.SelectedItem != null) {
                foreach (var ext in UsedExtensions) {
                    if (ext.ToString().Equals(LBUsed.SelectedItem.ToString())) {
                        AvaliableExtensions.Add(ext);
                        UsedExtensions.Remove(ext);
                        break;
                    }
                }
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
                bool shouldAdd = true;
                foreach (FileExtensionClass obj in AvaliableExtensions) {
                    if (obj.Extension.Equals(TextBox_Extension.Text))
                        shouldAdd = false;
                }
                foreach (FileExtensionClass obj in UsedExtensions) {
                    if (obj.Extension.Equals(TextBox_Extension.Text))
                        shouldAdd = false;
                }
                if (shouldAdd)
                    AvaliableExtensions.Add(new FileExtensionClass(TextBox_Extension.Text));

                TextBox_Extension.Text = "";
            }
            if (e.Key == Key.Escape)
                MyBaseWindow.Close();
        }

        private void LBAvaliable_KeyDown(object sender, KeyEventArgs e) {
            if (Key.Delete == e.Key) {
                List<FileExtensionClass> toDelete = new List<FileExtensionClass>();
                foreach (FileExtensionClass listViewItem in ((ListView)sender).SelectedItems) {
                    toDelete.Add(listViewItem);
                }
                foreach(var item in toDelete) {
                    AvaliableExtensions.Remove(item);
                }
            }
        }

        private void LBUsed_KeyDown(object sender, KeyEventArgs e) {
            if (Key.Delete == e.Key) {
                List<FileExtensionClass> toDelete = new List<FileExtensionClass>();
                foreach (FileExtensionClass listViewItem in ((ListView)sender).SelectedItems) {
                    toDelete.Add(listViewItem);
                }
                foreach (var item in toDelete) {
                    UsedExtensions.Remove(item);
                }
            }
        }

        private void UserControl_KeyDown(object sender, KeyEventArgs e) {
            if (e.Key == Key.Escape)
                MyBaseWindow.Close();
        }

        private void UserControl_Loaded(object sender, RoutedEventArgs e) {
            TextBox_Extension.Focus();
        }
    }
}
