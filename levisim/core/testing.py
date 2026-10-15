import time
from datetime import datetime
# NOTE: plyer gives desktop notifications and is purely a convenience, so it is optional.
# (it requires the pyobjc and pyobjus libraries too)
try:
    from plyer import notification
except ImportError:
    notification = None
from pprint import pprint
from levisim.core.logger import logger


__all__ = [
    'Test',
]


class Test:
    """
    Class that helps with a custom testing framework
    """
    def __init__(self, name, **kwargs):
        self.name = name
        self.exceptions = []

        self.start_title = kwargs.get('start_title', 'levisim test > 🏎️ START 💨')
        self.end_title = kwargs.get('end_title', 'levisim test > 🏁 END 🏁')

        # e.g. 'Mon 1 Jan 12:34:56:789012'
        self.datetime_format_string = kwargs.get('datetime_format_string', '%a %d %b @ %H:%M:%S.%f')
        self.datetime_string = lambda datetime_object:  datetime_object.strftime(self.datetime_format_string)

    def notify_start(self):
        """
        Notify this device that the test has started
        """
        self.start_time = time.time()
        self.start_datetime = datetime.fromtimestamp(self.start_time)
        self.start_message = f'"{self.name}"\n⏳: {self.datetime_string(self.start_datetime)}'
        logger.info(self.start_message)
        print(
            f'####################################################################################################'
            f'\n{self.start_title}: {self.start_message}\n'
            f'____________________________________________________________________________________________________'
        )
        if notification is not None:
            notification.notify(
                title=self.start_title,
                message=self.start_message,
                timeout=10,
            )

    def notify_end(self):
        """
        Notify this device that the test has ended
        """
        self.end_time = time.time()
        self.end_datetime = datetime.fromtimestamp(self.end_time)
        self.end_message = f'"{self.name}"\n⌛: {self.datetime_string(self.end_datetime)}'

        elapsed_seconds = self.end_time - self.start_time
        minutes, remaining_seconds = elapsed_seconds // 60, elapsed_seconds % 60

        if self.exceptions:
            self.end_message += f'\n❌ FAILED! 😔 {len(self.exceptions)} exception(s) raised'
        else:
            self.end_message += f'\n❇️ PASSED! 🥳'

        if elapsed_seconds < 1:
            time_taken = f'⏱️ Duration: {elapsed_seconds:.6f}s'
        if elapsed_seconds < 10:
            time_taken = f'⏱️ Duration: {elapsed_seconds:.3f}s'
        elif elapsed_seconds < 60:
            time_taken = f'⏱️ Duration: {elapsed_seconds:.2f}s'
        else:
            time_taken = f'⏱️ Duration: {minutes:.0f}m {remaining_seconds:.1f}s'

        self.end_message += f'\n{time_taken}'
        logger.info(self.end_message)
        print(
            f'____________________________________________________________________________________________________'
            f'\n{self.end_title}: {self.end_message}'
        )
        if 0 < len(self.exceptions) <= 10:
            print(f'Test "{self.name}" has exceptions: {self.exceptions}')
            logger.error(f'Test "{self.name}" has exceptions: {self.exceptions}')


        if notification is not None:
            notification.notify(
                title=self.end_title,
                message=self.end_message,
                timeout=10,
            )
