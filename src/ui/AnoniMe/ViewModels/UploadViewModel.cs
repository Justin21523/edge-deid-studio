using AnoniMe.Models;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using Microsoft.UI.Xaml;
using Microsoft.UI.Xaml.Controls;
using System;
using System.Collections.ObjectModel;
using System.Linq;
using System.Threading.Tasks;
using Windows.ApplicationModel.DataTransfer;
using Windows.Storage;
using Windows.Storage.Pickers;

namespace AnoniMe.ViewModels
{
    public partial class UploadViewModel : ObservableObject
    {
        public ObservableCollection<UploadFileInfo> UploadItems { get; } = new();
        public ObservableCollection<string> TagItems { get; } = new();

        [ObservableProperty] 
        private bool isUploading;

        [ObservableProperty] 
        private string? currentUploadText;

        [ObservableProperty] 
        private double currentUploadProgress;

        public Visibility UploadVisibility => IsUploading ? Visibility.Visible : Visibility.Collapsed;

        [RelayCommand]
        public async Task PickFilesAsync()
        {
            var picker = new FileOpenPicker();
            picker.FileTypeFilter.Add("*");

            var hwnd = WinRT.Interop.WindowNative.GetWindowHandle(App.MainAppWindow);
            WinRT.Interop.InitializeWithWindow.Initialize(picker, hwnd);

            var files = await picker.PickMultipleFilesAsync();
            if (files != null)
            {
                foreach (var file in files)
                    AddUploadItem(file);
            }
        }

        [RelayCommand]
        public async Task DropFilesAsync(DragEventArgs e)
        {
            if (e.DataView.Contains(StandardDataFormats.StorageItems))
            {
                var items = await e.DataView.GetStorageItemsAsync();
                foreach (var item in items.OfType<StorageFile>())
                    AddUploadItem(item);
            }
        }

        [RelayCommand]
        public void RemoveTag(string fileName)
        {
            var item = UploadItems.FirstOrDefault(f => f.FileName == fileName);
            if (item != null)
                UploadItems.Remove(item);

            TagItems.Remove(fileName);
        }

        [RelayCommand]
        public async Task GenerateResultAsync()
        {
            ContentDialog dialog = new()
            {
                Title = "生成中",
                Content = "將根據你選取的檔案與選項進行處理。",
                CloseButtonText = "OK",
                XamlRoot = App.MainAppWindow.Content.XamlRoot
            };
            await dialog.ShowAsync();
        }

        private void AddUploadItem(StorageFile file)
        {
            var item = new UploadFileInfo
            {
                FileName = file.Name,
                UploadProgress = 0,
                UploadStatus = "Waiting"
            };

            UploadItems.Add(item);
            AddFileTag(item.FileName);
            _ = SimulateUpload(item);
        }

        private void AddFileTag(string fileName)
        {
            if (!TagItems.Contains(fileName))
                TagItems.Add(fileName);
        }

        private async Task SimulateUpload(UploadFileInfo item)
        {
            IsUploading = true;
            CurrentUploadText = $"{item.FileName} uploading...";
            item.UploadStatus = "Uploading";

            for (int i = 0; i <= 100; i += 5)
            {
                await Task.Delay(50);
                item.UploadProgress = i;
                CurrentUploadProgress = i;
            }

            item.UploadStatus = "Upload complete";
            IsUploading = false;
        }


    }
}