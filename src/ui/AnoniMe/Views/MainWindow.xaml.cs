using System;
using System.Collections.ObjectModel;
using System.Linq;
using System.Threading.Tasks;
using Microsoft.UI.Xaml;
using Microsoft.UI.Xaml.Controls;
using Windows.Storage;
using Windows.Storage.Pickers;
using Windows.ApplicationModel.DataTransfer;
using Microsoft.UI.Xaml.Input;
using AnoniMe.ViewModels;
using Microsoft.Extensions.DependencyInjection;

namespace AnoniMe
{
    public sealed partial class MainWindow : Window
    {
        public HomeViewModel ViewModel { get; }
        public MainWindow()
        {
            this.InitializeComponent();

            // 從 DI 容器取得 MainViewModel 實例
            ViewModel = App.HostContainer.Services.GetService<HomeViewModel>();

            // 載入 HomePage 當首頁
            MainFrame.Navigate(typeof(HomePage), ViewModel);

        }
    }

}
