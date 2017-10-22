using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Controls;

namespace Zniffer
{
    [Serializable()]
    public class FileExtensionClass : INotifyPropertyChanged {
        private string _extension;

        public FileExtensionClass(string extension) {
            this._extension = extension;
        }

        public string Extension {
            get {
                return _extension;
            }
            set {
                if (_extension != value) {
                    _extension = value;
                    OnPropertyChanged("ext");
                }
            }
        }


        private void OnPropertyChanged(string propertyName) {
            PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(propertyName));
        }

        public event PropertyChangedEventHandler PropertyChanged;

        public override string ToString() {
            return _extension;
        }
    }
}
