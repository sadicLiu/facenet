import os

path = '/home/liuhy/Downloads'
with open(os.path.join(path, 'result.txt'), 'at') as f:
    f.write('test')
    f.flush()
    f.close()