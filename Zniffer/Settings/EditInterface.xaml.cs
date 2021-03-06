﻿using System.Windows;
using System.Windows.Controls;
using System.Windows.Input;

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
            strIP.Text = interfaceObj.Addres;
            strPort.Text = interfaceObj.Ports;

        }

        private void MyBaseWindow_SizeChanged(object sender, SizeChangedEventArgs e) {
            Height = MyBaseWindow.Height;
            Width = MyBaseWindow.Width;
        }

        public string getPorts() {
            return strPort.Text;
        }

        private void strPort_KeyDown(object sender, KeyEventArgs e) {
            if(e.Key == Key.Enter || e.Key == Key.Escape)
                MyBaseWindow.Close();
        }

        private void EditIFace_Loaded(object sender, RoutedEventArgs e) {
            strPort.Focus();

        }
    }
}
