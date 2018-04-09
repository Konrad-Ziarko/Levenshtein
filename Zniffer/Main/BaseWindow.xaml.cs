using System;
using System.Windows;

namespace Zniffer {
    /// <summary>
    /// Interaction logic for BaseWindow.xaml
    /// </summary>
    public partial class BaseWindow : Window {


        public void Window_SourceInitialized(object sender, EventArgs e) {

        }

        private void Window_Closing(object sender, System.ComponentModel.CancelEventArgs e) {
            Properties.Settings.Default.Save();
        }

        #region TitleBar buttons
        private void button_Exit_Click(object sender, RoutedEventArgs e) {
            Close();
        }

        private void button_Max_Click(object sender, RoutedEventArgs e) {
            if (WindowState == WindowState.Maximized)
                WindowState = WindowState.Normal;
            else
                WindowState = WindowState.Maximized;
        }

        private void button_Min_Click(object sender, RoutedEventArgs e) {
            WindowState = WindowState.Minimized;
        }
        #endregion
        public BaseWindow() {
            InitializeComponent();
        }
    }
}
