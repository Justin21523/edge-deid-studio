using AnoniMe.Models;
using AnoniMe.Views;
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

        [ObservableProperty] private bool isUploading;
        [ObservableProperty] private string? currentUploadText;
        [ObservableProperty] private double currentUploadProgress;

        private bool isProcessing;
        public bool IsProcessing
        {
            get => isProcessing;
            set => SetProperty(ref isProcessing, value);
        }

        // 遮蔽選項
        private bool maskIdentity;
        public bool MaskIdentity
        {
            get => maskIdentity;
            set
            {
                SetProperty(ref maskIdentity, value);
                OnPropertyChanged(nameof(CanGenerate));
            }
        }

        private bool maskAddress;
        public bool MaskAddress
        {
            get => maskAddress;
            set
            {
                SetProperty(ref maskAddress, value);
                OnPropertyChanged(nameof(CanGenerate));
            }
        }

        // ✅ 是否可以按下生成按鈕
        public bool CanGenerate => TagItems.Any() && (MaskIdentity || MaskAddress);

        public Visibility UploadVisibility => IsUploading ? Visibility.Visible : Visibility.Collapsed;

        public UploadViewModel()
        {
            // TagItems 改變時同步通知 UI 更新 CanGenerate
            TagItems.CollectionChanged += (_, __) => OnPropertyChanged(nameof(CanGenerate));
        }

        [RelayCommand]
        public async Task GenerateResultAsync()
        {
            if (!CanGenerate)
            {
                ContentDialog warning = new()
                {
                    Title = "請檢查輸入條件",
                    Content = "請選擇至少一個檔案，並勾選一項遮蔽選項後才能生成結果。",
                    CloseButtonText = "我知道了",
                    XamlRoot = App.MainAppWindow.Content.XamlRoot
                };
                await warning.ShowAsync();
                return;
            }

            // 顯示處理中遮罩
            IsProcessing = true;
            await Task.Delay(2000); // 模擬處理時間（可換成實際處理邏輯）
            IsProcessing = false;

            // 導向 ResultPage
            if (App.MainAppWindow.Content is Frame frame)
            {
                frame.Navigate(typeof(ResultPage));
            }
        }

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

        public void HandleDroppedFile(StorageFile file)
        {
            AddFileTag(file.Name); // ✨ 加入檔名做為 tag
        }

        [RelayCommand]
        public void RemoveTag(string fileName)
        {
            var item = UploadItems.FirstOrDefault(f => f.FileName == fileName);
            if (item != null)
                UploadItems.Remove(item);

            TagItems.Remove(fileName);
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