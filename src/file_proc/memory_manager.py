import resource

class EdgeMemoryGuard:
    def __enter__(self):
        # 設定記憶體上限為 512MB
        resource.setrlimit(
            resource.RLIMIT_AS,
            (512 * 1024 * 1024, resource.RLIM_INFINITY)
        )
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # TODO: cleanup if necessary
        pass
