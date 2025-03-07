import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict

from tools.extract_features import VitFeatureExtractor, DinoV2FeatureExtractor
from tools.utils import convert_bytes_to_image

# 设置转字符串的最大长度，防止自动截断
np.set_printoptions(threshold=5000)


def add_prev_data(dir_path: str = './images', target_file_path: str = './tools/embeddings/icon_embeddings_prev.csv'):
    """
    增加预先提取的数据到数据表中
    :param dir_path: 数据文件夹 
        - dir_path
            - class1
                - file1
                - file2
            - class2
                - file1
    :param target_file_path: 存储的csv文件路径
    :return: 
    """
    extractor_dict = {
        'vit': VitFeatureExtractor,
        'dinov2': DinoV2FeatureExtractor
    }
    add_data_dict = defaultdict(list)
    
    for class_dir_path in Path(dir_path).glob('*'):
        if class_dir_path.is_dir():
            class_name = class_dir_path.name
            for img_path in class_dir_path.glob('*'):
                if img_path.suffix in ['.jpg', '.jpeg', '.png']:
                    with open(img_path, 'rb') as f:
                        image_data_buf = f.read()
                    image_data = convert_bytes_to_image(image_data_buf)
                    
                    for extract_func in extractor_dict.keys():
                        extractor = extractor_dict[extract_func]()
                        # 转字符串 方便后续合并去重
                        feature = np.array2string(extractor.extract(image_data))
                        add_data_dict[extract_func].append(feature)
                    add_data_dict['class'].append(class_name)
                    
    add_data_df = pd.DataFrame(add_data_dict)

    # 如果存在之前的数据，则去重并合并
    if Path(target_file_path).exists():
        cur_df = pd.read_csv(target_file_path)
        add_data_df = pd.concat([cur_df, add_data_df])
        add_data_df.drop_duplicates(inplace=True)
    add_data_df.to_csv(target_file_path, index=False)
                    

if __name__ == '__main__':
    add_prev_data()


