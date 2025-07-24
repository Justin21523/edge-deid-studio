using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using Microsoft.UI.Xaml.Controls;
using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace AnoniMe.ViewModels
{

    public partial class ResultViewModel : ObservableObject
    {
        public ObservableCollection<string> ProcessedFileNames { get; } = new()
        {
            "report_masked.pdf",
            "notes_cleaned.txt"
        };

        [ObservableProperty]
        private string filePreviewText = "這是預覽內容，模擬網頁展示效果。";

        [RelayCommand]
        public void Download() { /* 實作下載邏輯 */ }

        [RelayCommand]
        public void GoHome()
        {
            if (App.MainAppWindow?.Content is Frame rootFrame)
            {
                rootFrame.Navigate(typeof(HomePage));
            }
        }

    }
}
