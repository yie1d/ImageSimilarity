import re
import faiss

import numpy as np
import pandas as pd
import typing as t
from io import BytesIO

from PIL import Image


def convert_bytes_to_image(image_data_buf: bytes) -> Image.Image:
    """将图像bytes转换为RGB图像数据"""
    image_buffer = BytesIO()
    image_buffer.write(image_data_buf)
    image_buffer.seek(0)
    image = Image.open(image_buffer)
    if image.mode != 'RGB':
        image = image.convert('RGB')
    return image


def clean_feature_string(feature_str: str) -> np.ndarray:
    """将从csv文件中读取的特征字符串转换为numpy数组"""
    cleaned_str = re.sub(r'[\[\]]', '', feature_str)
    cleaned_values = np.fromstring(cleaned_str, sep=' ')
    return cleaned_values


def get_top_k_similar(input_embedding: np.ndarray, database_features: np.ndarray, k=5) -> t.List[t.Tuple[int, float]]:
    """通过FAISS计算最相似的K个结果"""
    # 归一化输入向量
    input_normalized = input_embedding / np.linalg.norm(input_embedding, axis=1, keepdims=True)
    input_normalized = input_normalized.reshape(1, -1)

    # 归一化特征向量的余弦相似度
    database_normalized = database_features / np.linalg.norm(database_features, axis=1, keepdims=True)

    # 内积距离
    faiss_index = faiss.IndexFlatIP(database_features.shape[1])
    faiss_index.add(database_normalized)

    # 搜索前K个最相似的结果
    distances, top_k_idx = faiss_index.search(input_normalized, k)

    # 返回前k个结果的索引和相似度
    return [(idx, distances[0][inx]) for inx, idx in enumerate(top_k_idx[0])]


def classify_by_features(
        input_embedding: np.ndarray,
        database_embeddings_df: pd.DataFrame,
        extract_func: str = 'dinov2'
) -> t.Tuple[t.Optional[str], float]:
    """通过预先生成的特征向量文件对目标进行分类"""
    features = np.array([clean_feature_string(_) for _ in database_embeddings_df[extract_func].values])
    top_similar = get_top_k_similar(input_embedding, features, k=1)[0]
    if not top_similar:
        return None, 0

    return database_embeddings_df['class'][top_similar[0]], top_similar[1]
