import os

DATA_DIRECTORY = "data"

def working_directory():
    """
     get current directory
    """
    return os.path.join(os.getcwd(), DATA_DIRECTORY)

def read_file_lines(dataset, filename):
    """
    read all lines of file with file name, not full path
    """
    filepath = os.path.join(
    working_directory(), dataset, filename)
    with open(filepath, 'r', encoding='utf-8') as content:
        return content.readlines()
    


