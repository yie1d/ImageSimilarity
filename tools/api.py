import base64
import logging
import pandas as pd
import typing as t
from pathlib import Path

from tools.extract_features import VitFeatureExtractor, DinoV2FeatureExtractor
from tools.utils import convert_bytes_to_image, classify_by_features
from tools.exceptions import RequestParamsException
from tools.logger import get_logger


class ImageSimilarityInference:
    def __init__(self, logger_obj: t.Optional[logging.Logger] = None):
        self.vit_extractor = VitFeatureExtractor()
        self.dinov2_extractor = DinoV2FeatureExtractor()
        self.prev_embeddings_df = pd.read_csv(str(Path(__file__).parent.joinpath('embeddings').joinpath('icon_embeddings_prev.csv')))

        if logger_obj is None:
            logger_obj = get_logger(__name__, Path(__file__).parent.parent.joinpath('logs'))
        self.logger = logger_obj

    def inference(self, req_data: dict) -> t.Dict[str, t.Dict[str, t.Union[str, float]]]:
        """图标分类"""
        extract_func = req_data.get('extract_func', 'dinov2').lower()
        if extract_func not in ['dinov2', 'vit']:
            raise RequestParamsException('请求参数extract_func异常')

        image_data_buf_dict = req_data.get('images')

        if image_data_buf_dict is None or isinstance(image_data_buf_dict, dict) is False:
            raise RequestParamsException('请求参数images字段异常')

        try:
            image_classify_dict = {}
            for image_id, image_data_buf in image_data_buf_dict.items():
                image_data_buf = image_data_buf.encode("utf-8")
                image_data_buf = base64.b64decode(image_data_buf)

                if extract_func == 'vit':
                    feature = self.vit_extractor.extract(convert_bytes_to_image(image_data_buf))
                else:
                    feature = self.dinov2_extractor.extract(convert_bytes_to_image(image_data_buf))

                _class, _score = classify_by_features(feature, self.prev_embeddings_df, extract_func=extract_func)
                # 如果这里需要过滤返回结果，建议阈值使用0.65
                image_classify_dict[image_id] = {
                    'class': _class,
                    'score': f'{_score:.3f}'
                }
        except Exception as e:
            self.logger.info(f'图标分类异常: {e}')
            raise RequestParamsException('images解析异常')

        return image_classify_dict


if __name__ == '__main__':
    with open('../examples/1.png', 'rb') as f:
        _image_data_buf = f.read()
    print(ImageSimilarityInference().inference({
        'extract_func': 'dinov2',
        'images': {
            '1': base64.b64encode(_image_data_buf).decode('utf-8')
        }
    }))
