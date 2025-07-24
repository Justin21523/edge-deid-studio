using CommunityToolkit.Mvvm.Input;


namespace AnoniMe.Models
{
    public class PreviewFileInfo
    {
        public string FileName { get; set; }
        public string FileType { get; set; }
        public string FileSizeDisplay { get; set; }
        public string PreviewText { get; set; }
        public string? PdfUri { get; set; }
        public bool IsPdf { get; set; }
        public bool IsDoc { get; set; }
    }
}
