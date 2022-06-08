

def headline(message):
    buffer_len = (80 - len(message))//2 if len(message) < 80 else 0
    return "-"*buffer_len + ' ' + message + ' ' + '-'*buffer_len