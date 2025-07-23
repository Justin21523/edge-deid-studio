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

            // �q DI �e�����o MainViewModel ���
            ViewModel = App.HostContainer.Services.GetService<HomeViewModel>();

            // ���J HomePage ����
            MainFrame.Navigate(typeof(HomePage), ViewModel);

        }
    }

}
