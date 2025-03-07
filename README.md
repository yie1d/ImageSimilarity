# ImageSimilarity
通过VIT及Dino_V2获取图像的embeddings，再通过faiss计算目标图像与已知图像的距离，得到最相似的结果。

-------------------------
## 1. Project structure
```python
- examples
    - 1.png  # 用于举例的图片
    - 2.png  # 用户举例的图片
    - example1  # 示例1，仅调用的示例
    - example2  # 示例2，作为服务时的调用示例
- images  # 用于存放计算图像向量的文件目录
    - class1  # 类别名1
        - img1.png
        - img2.png
    - class2  # 类别名2
        - img1.png
        - img2.png
- logs  # 存放日志文件
- tools  # 用于计算推理的方法
    - embeddings  # 存放用于计算距离的向量结果的文件夹
    - weights  # 存放模型预训练权重
        - dinov2-base
        - vit-large-patch16-224-in21k
    - api.py  # 相似度推理器
    - exceptions  # 自定义异常
    - extract_features  # embeddings提取器
    - logger.py  # 自定义日志记录器
    - utils  # 公共基础方法，如计算向量相似度等
- add_prev_data.py  # 增加用于比较的图像向量的脚本
- server.py  # 用于启动本地接口服务
```

## 2. Install

#### 2.1 如果使用conda，首先创建环境（已激活环境或不使用虚拟环境的跳过此步）
```python
conda create -n image_similarity_env python==3.8
conda activate image_similarity_env
```

#### 2.2 创建环境后或者直接使用当前环境
```python
cd ImageSimilarity
# 如果是cpu运行
pip install -r requirements-cpu.txt
# 如果是gpu运行
pip install -r requirements-gpu.txt
```

#### 2.3 下载模型预训练权重
下载权重文件 "model.safetensors" 分别放到 "./tools/weights/对应的文件夹/" 下
##### vit 权重
```python
https://huggingface.co/google/vit-large-patch16-224-in21k/tree/main
或者
https://hf-mirror.com/google/vit-large-patch16-224-in21k/tree/main
```

##### dino_v2 权重
```python
https://huggingface.co/facebook/dinov2-base/tree/main
或者
https://hf-mirror.com/facebook/dinov2-base/tree/main
```

## 3. Examples
- example1: 直接调用
```python
python examples/example1.py
```

- example2: 作为服务
```python
python server.py
python examples/example2.py
```

## 4. Generate images for comparison
首先准备作为缓存图像
再执行脚本
```python
python add_prev_data.py
```