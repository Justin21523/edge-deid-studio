import psutil

class ResourceMonitor:
    @staticmethod
    def current_load() -> float:
      return psutil.cpu_percent()

    @staticmethod
    def adjust_processing(config: dict) -> dict:
        load = ResourceMonitor.current_load()
        if load > 80:
            config['ocr']['enable'] = False
            config['face_detection']['min_confidence'] = 0.9
        elif load > 60:
            config['text_detection']['fast_mode'] = True
        return config
