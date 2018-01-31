using System;
using System.Windows.Controls;
using System.Windows.Data;

namespace Zniffer {
    public class Converter : IValueConverter
    {
        public object Convert(object value, Type targetType, object parameter, System.Globalization.CultureInfo culture)
        {
            ItemsControl ic = value as ItemsControl;
            return ic.ActualHeight / ic.Items.Count;
        }

        public object ConvertBack(object value, Type targetType, object parameter, System.Globalization.CultureInfo culture)
        {
            throw new NotImplementedException();
        }
    }
}
