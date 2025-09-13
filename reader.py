from utils import xp

class SlidingComplex64Reader:
    dtype = xp.complex64
    itemsize = 8  # complex64

    def __init__(self, file_path):
        """
        Initializes the reader for a complex64 binary file.

        Args:
            file_path (str): The path to the binary file.
        """
        self.file_path = file_path

    def get(self, start, length):
        """
        Reads a chunk of data from the file.

        Args:
            start (int): The starting index (in terms of complex64 elements).
            length (int): The number of complex64 elements to read.

        Returns:
            xp.ndarray: An array containing the requested data.
        """
        byte_start = start * self.itemsize
        byte_length = length * self.itemsize

        try:
            with open(self.file_path, 'rb') as f:
                f.seek(byte_start)
                data_bytes = f.read(byte_length)

            if len(data_bytes) != byte_length:
                raise IOError(f"Read fewer bytes than expected. Expected {byte_length}, got {len(data_bytes)}")

            # Convert the bytes to a NumPy/CuPy array
            # Note: The .frombuffer method is efficient as it creates a view of the bytes.
            # We then copy it to the appropriate device (CPU or GPU) if necessary.
            data_array = xp.frombuffer(data_bytes, dtype=self.dtype)
            return data_array

        except FileNotFoundError:
            print(f"Error: The file '{self.file_path}' was not found.")
            return None
        except Exception as e:
            print(f"An error occurred: {e}")
            return None