import resource

class EdgeMemoryGuard:
    def __enter__(self):
        # TODO: set memory limit
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # TODO: cleanup if necessary
        pass
