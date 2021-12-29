from datetime import datetime

def currentdate():
    return datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
