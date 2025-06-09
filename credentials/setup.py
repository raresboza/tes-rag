import os


def read_credentials(file_path):
    """Read API key from text file"""
    with open(file_path, 'r') as file:
        return file.read().strip()


def setup_credentials():
    api_key = read_credentials("google_api_key.txt")
    os.environ["GOOGLE_API_KEY"] = api_key