import signal
import threading

class DelayedInterrupt:
    """Delay Ctrl+C only if running in the main thread."""

    def __enter__(self):
        self.signal_received = None
        self.is_main = threading.current_thread() is threading.main_thread()
        if self.is_main:
            self.old_handler = signal.getsignal(signal.SIGINT)
            signal.signal(signal.SIGINT, self.handler)
        return self

    def handler(self, sig, frame):
        self.signal_received = (sig, frame)
        print("\nCtrl+C pressed — will interrupt after current save...")

    def __exit__(self, exc_type, exc_value, traceback):
        if self.is_main:
            signal.signal(signal.SIGINT, self.old_handler)
            if self.signal_received:
                self.old_handler(*self.signal_received)


class DelayedInterruptMainProcess:
    def __enter__(self):
        self.signal_received = None
        self.old_handler = signal.getsignal(signal.SIGINT)
        signal.signal(signal.SIGINT, self.handler)
        return self

    def handler(self, sig, frame):
        # Store the signal — don't interrupt yet
        self.signal_received = (sig, frame)
        print("\nCtrl+C pressed. Will interrupt after current task...")

    def __exit__(self, type, value, traceback):
        # Restore old handler
        signal.signal(signal.SIGINT, self.old_handler)
        # If Ctrl+C was pressed, re-raise KeyboardInterrupt now
        if self.signal_received:
            self.old_handler(*self.signal_received)