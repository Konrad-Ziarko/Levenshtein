using System;
using System.Windows;
using Zniffer.Network;

namespace Zniffer {
    /// <summary>
    /// Interaction logic for MainWindow.xaml
    /// </summary>
    public partial class MainWindow : Window {

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

        public MainWindow() {

            Sniffer snf = new Sniffer();
        }

    }
}
