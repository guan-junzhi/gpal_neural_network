class DataloaderException(Exception):
    """This exception is used for dataloading related issue, it should stop the running process"""


class DatasetException(Exception):
    """This exception is used when the initialization of user customized dataset."""


class DataSkipException(Exception):
    """This exception is used to indentify the data in skipping list"""

