import pathlib


def check_and_mkdir(path):
    path = pathlib.Path(path).parent
    if path.exists() or path == '/':
        return
    check_and_mkdir(path)
    if not path.exists():
        path.mkdir()