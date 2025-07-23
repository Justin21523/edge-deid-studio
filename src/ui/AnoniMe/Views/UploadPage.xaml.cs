using Microsoft.Extensions.DependencyInjection;
using Microsoft.UI.Xaml;
using Microsoft.UI.Xaml.Controls;
using Microsoft.UI.Xaml.Input;
using System;
using System.Collections.ObjectModel;
using System.Linq;
using System.Threading.Tasks;
using Windows.ApplicationModel.DataTransfer;
using Windows.Storage;
using Windows.Storage.Pickers;

namespace AnoniMe
{
    public sealed partial class UploadPage : Page
    {
        public ViewModels.UploadViewModel ViewModel { get; } = App.HostContainer.Services.GetService<ViewModels.UploadViewModel>();

        public UploadPage()
        {
            this.InitializeComponent();
            this.DataContext = ViewModel;
        }

       
        private void DropZone_DragOver(object sender, DragEventArgs e)
        {
            e.AcceptedOperation = DataPackageOperation.Copy;
            e.DragUIOverride.Caption = "放開以上傳";
            e.DragUIOverride.IsContentVisible = true;
        }

        private void DropZone_Drop(object sender, DragEventArgs e)
        {
            if (ViewModel.DropFilesCommand.CanExecute(e))
            {
                ViewModel.DropFilesCommand.Execute(e);
            }
        }
    }
}
