using CommunityToolkit.Mvvm.Input;


namespace AnoniMe.Models
{
    public class PreviewItem
    {
        public string FileName { get; set; } = "";
        public string PreviewContent { get; set; } = "";
        public IRelayCommand DownloadCommand { get; set; } = new RelayCommand(() => { });
    }
}
