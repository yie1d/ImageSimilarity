from flask import Flask, Response, jsonify, request

from tools.api import ImageSimilarityInference
from tools.logger import get_logger
from tools.exceptions import RequestParamsException


class ImageSimilarityServer:
    def __init__(self):
        self.server = Flask(__name__)

        self.server.route('/icon_classify', methods=['POST'])(self.icon_classify)
        self.logger = get_logger('image_similarity', './logs')
        infer = ImageSimilarityInference(self.logger)
        self.api = infer.inference

    def icon_classify(self) -> Response:
        """图标分类"""
        req_data = request.get_json()
        
        try:
            return jsonify({
                'code': '0000',
                'content': self.api(req_data)
            })
        except RequestParamsException as e:
            self.logger.error(f'icon_classify RequestParamsException: {e.description}')
            return jsonify({
                'code': '4000',
                'description': e.description
            })
        except Exception as e:
            self.logger.error(f'icon_classify: {e}')


if __name__ == '__main__':
    image_similarity_server = ImageSimilarityServer()
    image_similarity_server.server.run('0.0.0.0', 7863)
