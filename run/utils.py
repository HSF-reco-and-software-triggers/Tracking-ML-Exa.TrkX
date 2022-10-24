import os, shutil

def headline(message):
    buffer_len = (80 - len(message))//2 if len(message) < 80 else 0
    return "-"*buffer_len + ' ' + message + ' ' + '-'*buffer_len

def delete_directory(dir):
    if os.path.isdir(dir):
        for files in os.listdir(dir):
            path = os.path.join(dir, files)
            try:
                shutil.rmtree(path)
            except OSError:
                os.remove(path)