class BASEException(Exception):
    """Base class for exceptions in this module."""
    pass

class FileNotFoundError(BASEException):
    """Exception raised for errors in the input file not found."""
    def __init__(self, filepath):
        self.message = f"The file {filepath} does not exist."
        super().__init__(self.message)

class EmptyFileError(BASEException):
    """Exception raised for empty files."""
    def __init__(self, filepath):
        self.message = f"The file {filepath} is empty."
        super().__init__(self.message)

class InvalidSequenceError(BASEException):
    """Exception raised for invalid DNA or protein sequences."""
    def __init__(self, header, sequence):
        self.message = f"Invalid sequence detected under header {header}: {sequence}"
        super().__init__(self.message)

class GeneralFASTAError(BASEException):
    """Exception raised for general errors in this module."""
    def __init__(self, filepath, error):
        self.message = f"An error occurred while reading the file {filepath}: {error}"
        super().__init__(self.message)