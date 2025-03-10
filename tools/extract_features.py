import torch
import numpy as np

from pathlib import Path
from abc import ABC, abstractmethod
from PIL import Image
from transformers import AutoImageProcessor, ViTModel, AutoModel, image_processing_base

from tools.utils import convert_bytes_to_image
from tools.exceptions import ModelNotFoundException


class FeatureExtractor(ABC):
    image_processor = None
    model = None
    MODEL_WEIGHTS_PATH = str(Path(__file__).parent.joinpath('weights'))
    INSTANCE = None

    def __new__(cls, *args, **kwargs):
        if cls.INSTANCE is None:
            cls.INSTANCE = super().__new__(cls, *args, **kwargs)
            cls._init_model(cls.INSTANCE)
        return cls.INSTANCE
    
    @abstractmethod
    def _init_model(self):
        """初始化模型及图像处理器"""
        raise NotImplementedError("This method should be overridden.")

    def get_embedding(self, inputs: image_processing_base.BatchFeature) -> np.ndarray:
        with torch.no_grad():
            outputs = self.model(**inputs)
        embedding = outputs.last_hidden_state
        embedding = embedding[:, 0, :].squeeze(1)
        return embedding.numpy()

    def extract(self, img_data: Image.Image) -> np.ndarray:
        inputs = self.image_processor(img_data, return_tensors='pt')

        return self.get_embedding(inputs)
    
    
class VitFeatureExtractor(FeatureExtractor):
    MODEL_NAME = 'vit-large-patch16-224-in21k'
    
    def _init_model(self):
        pretrained_model_path = Path(self.MODEL_WEIGHTS_PATH).joinpath(self.MODEL_NAME)
        safetensors_path = pretrained_model_path.joinpath('model.safetensors')
        if safetensors_path.exists() is False:
            raise ModelNotFoundException(f'{safetensors_path}文件不存在，下载地址：https://hf-mirror.com/google/vit-large-patch16-224-in21k/tree/main')
        self.image_processor = AutoImageProcessor.from_pretrained(pretrained_model_path, use_fast=True)
        self.model = ViTModel.from_pretrained(pretrained_model_path)

        
class DinoV2FeatureExtractor(FeatureExtractor):
    MODEL_NAME = 'dinov2-base'
    
    def _init_model(self):
        pretrained_model_path = Path(self.MODEL_WEIGHTS_PATH).joinpath(self.MODEL_NAME)
        safetensors_path = pretrained_model_path.joinpath('model.safetensors')
        if safetensors_path.exists() is False:
            raise ModelNotFoundException(f'{safetensors_path}文件不存在，下载地址：https://hf-mirror.com/facebook/dinov2-base/tree/main')
        self.image_processor = AutoImageProcessor.from_pretrained(pretrained_model_path)
        self.model = AutoModel.from_pretrained(pretrained_model_path)


if __name__ == '__main__':
    import pandas as pd
    from tools.utils import classify_by_features

    extract_func = 'vit'
    extractor = VitFeatureExtractor()
    with open('../examples/1.png', 'rb') as f:
        image_data_buf = f.read()
    feature = extractor.extract(convert_bytes_to_image(image_data_buf))
    embeddings_df = pd.read_csv('./embeddings/icon_embeddings_prev.csv')
    _class, _score = classify_by_features(feature, database_embeddings_df=embeddings_df, extract_func=extract_func)
    print(_class, _score)
    