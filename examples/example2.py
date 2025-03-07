import base64
import requests
from pathlib import Path


if __name__ == '__main__':
    # !! 需要先启动server.py
    with open(Path(__file__).parent.joinpath('1.png'), 'rb') as f:
        file_data = f.read()

    with open(Path(__file__).parent.joinpath('2.png'), 'rb') as f:
        file_data1 = f.read()

    response = requests.post(
        'http://127.0.0.1:7863/icon_classify',
        json={
            'extract_func': 'vit',
            'images': {
                1: base64.b64encode(file_data).decode('utf-8'),
                2: base64.b64encode(file_data1).decode('utf-8')
            },
        },
        headers={
            'Content-Type': 'application/json'
        }
    )
    print(response.json())


