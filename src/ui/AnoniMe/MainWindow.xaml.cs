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

namespace AnoniMe
{
    public sealed partial class MainWindow : Window
    {
        public ObservableCollection<UploadFileInfo> UploadItems { get; } = new();
        public ObservableCollection<string> TagItems { get; } = new();

        public MainWindow()
        {
            this.InitializeComponent();
        }

        private async void PickFiles_Click(object sender, RoutedEventArgs e)
        {
            var picker = new FileOpenPicker();
            picker.FileTypeFilter.Add("*");
            var hwnd = WinRT.Interop.WindowNative.GetWindowHandle(this);
            WinRT.Interop.InitializeWithWindow.Initialize(picker, hwnd);
            var files = await picker.PickMultipleFilesAsync();

            if (files != null)
            {
                foreach (var file in files)
                {
                    AddUploadItem(file);
                }
            }
        }

        private async void DropZone_Drop(object sender, DragEventArgs e)
        {
            if (e.DataView.Contains(StandardDataFormats.StorageItems))
            {
                var items = await e.DataView.GetStorageItemsAsync();
                foreach (var item in items)
                {
                    if (item is StorageFile file)
                    {
                        AddUploadItem(file);
                    }
                }
            }
        }

        private void DropZone_DragOver(object sender, DragEventArgs e)
        {
            e.AcceptedOperation = DataPackageOperation.Copy;
            e.DragUIOverride.Caption = "放開以上傳";
            e.DragUIOverride.IsContentVisible = true;
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
            SimulateUpload(item);
        }

        private void AddFileTag(string fileName)
        {
            if (!TagItems.Contains(fileName))
            {
                TagItems.Add(fileName);
            }
        }

        private void RemoveTag_Click(object sender, RoutedEventArgs e)
        {
            if ((sender as Button)?.Tag is string fileName)
            {
                var itemToRemove = UploadItems.FirstOrDefault(f => f.FileName == fileName);
                if (itemToRemove != null)
                    UploadItems.Remove(itemToRemove);

                TagItems.Remove(fileName);
            }
        }

        private async void SimulateUpload(UploadFileInfo item)
        {
            UploadStatusPanel.Visibility = Visibility.Visible;
            CurrentUploadText.Text = $"{item.FileName} uploading...";
            item.UploadStatus = "Uploading";

            for (int i = 0; i <= 100; i += 5)
            {
                await Task.Delay(50);
                item.UploadProgress = i;
                CurrentUploadProgress.Value = i;
            }

            item.UploadStatus = "Upload complete";
            UploadStatusPanel.Visibility = Visibility.Collapsed;
        }

        private void Generate_Click(object sender, RoutedEventArgs e)
        {
            ContentDialog dialog = new ContentDialog
            {
                Title = "生成中",
                Content = "將根據你選取的檔案與選項進行處理。",
                CloseButtonText = "OK",
                XamlRoot = this.Content.XamlRoot
            };
            _ = dialog.ShowAsync();
        }

        private void RemoveItem_Click(object sender, RoutedEventArgs e)
        {
            if (sender is Button button && button.Tag is UploadFileInfo file)
            {
                UploadItems.Remove(file);
                TagItems.Remove(file.FileName); // 如果有加到 Tags 的話
            }
        }



    }
}
