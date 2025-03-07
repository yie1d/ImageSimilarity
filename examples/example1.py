import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import base64
from tools.api import ImageSimilarityInference

if __name__ == '__main__':
    with open(Path(__file__).parent.joinpath('1.png'), 'rb') as f:
        _image_data_buf1 = f.read()

    with open(Path(__file__).parent.joinpath('2.png'), 'rb') as f:
        _image_data_buf2 = f.read()

    infer = ImageSimilarityInference()
    print(infer.inference({
        'extract_func': 'dinov2',  # 'dinov2' or 'vit'
        'images': {
            '1': base64.b64encode(_image_data_buf1).decode('utf-8'),
            '2': base64.b64encode(_image_data_buf2).decode('utf-8')
        }
    }))
