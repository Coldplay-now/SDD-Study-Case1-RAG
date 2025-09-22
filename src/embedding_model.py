#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
嵌入模型模块
专门负责文本向量化处理
"""

import logging
import numpy as np
from typing import List
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

class EmbeddingModel:
    """
    嵌入模型类
    
    负责文本向量化处理，使用SentenceTransformer模型将文本转换为高维向量表示。
    支持批量处理和多种预训练模型，主要用于RAG系统中的文档和查询向量化。
    
    主要功能：
    - 加载和管理预训练的嵌入模型
    - 将文本转换为标准化的向量表示
    - 支持批量文本处理以提高效率
    - 自动处理模型加载和设备管理
    
    使用示例：
        >>> model = EmbeddingModel("BAAI/bge-small-zh-v1.5", "cpu")
        >>> model.load_model()
        >>> vectors = model.encode_texts(["文本1", "文本2"])
    """
    
    def __init__(self, model_name: str = "BAAI/bge-small-zh-v1.5", device: str = "cpu"):
        """
        初始化嵌入模型
        
        Args:
            model_name (str): 嵌入模型名称，支持HuggingFace模型库中的SentenceTransformer模型
                            默认使用BAAI/bge-small-zh-v1.5，适合中文文本处理
            device (str): 计算设备，可选值：
                         - "cpu": 使用CPU计算（默认）
                         - "cuda": 使用GPU计算（需要CUDA支持）
                         - "mps": 使用Apple Silicon GPU（macOS）
        
        Attributes:
            model_name (str): 模型名称
            device (str): 计算设备
            model (SentenceTransformer): 加载的模型实例，初始为None
            embedding_dim (int): 嵌入向量维度，模型加载后确定
        """
        self.model_name = model_name
        self.device = device
        self.model = None
        self.embedding_dim = None
        logger.info(f"嵌入模型初始化 - 模型: {model_name}, 设备: {device}")
    
    def load_model(self):
        """
        加载嵌入模型
        
        从HuggingFace模型库下载并加载指定的SentenceTransformer模型。
        首次使用时会自动下载模型文件到本地缓存，后续使用会直接从缓存加载。
        
        加载过程：
        1. 实例化SentenceTransformer模型
        2. 执行测试编码以确定向量维度
        3. 设置模型为评估模式
        
        Returns:
            bool: 加载成功返回True，失败返回False
            
        Raises:
            Exception: 模型下载失败、设备不支持或其他加载错误
            
        Note:
            - 首次加载可能需要较长时间下载模型文件
            - 确保网络连接正常以便下载模型
            - GPU设备需要安装对应的CUDA或MPS支持
        """
        try:
            logger.info(f"正在加载嵌入模型: {self.model_name}")
            self.model = SentenceTransformer(self.model_name, device=self.device)
            
            # 获取嵌入维度
            test_embedding = self.model.encode(["测试文本"])
            self.embedding_dim = test_embedding.shape[1]
            
            logger.info(f"模型加载成功，嵌入维度: {self.embedding_dim}")
            return True
            
        except Exception as e:
            logger.error(f"模型加载失败: {str(e)}")
            return False
    
    def encode_texts(self, texts: List[str]) -> np.ndarray:
        """
        编码文本为向量
        
        将输入的文本列表转换为标准化的向量表示。使用预加载的SentenceTransformer模型
        进行批量编码，自动处理文本预处理和向量标准化。
        
        处理流程：
        1. 验证模型是否已加载
        2. 批量编码所有输入文本
        3. 对向量进行L2标准化
        4. 返回标准化后的向量数组
        
        Args:
            texts (List[str]): 待编码的文本列表，每个元素为一个字符串
                              支持中英文混合文本，自动处理特殊字符
        
        Returns:
            np.ndarray: 标准化的向量数组，形状为 (len(texts), embedding_dim)
                       每行对应一个输入文本的向量表示
        
        Raises:
            ValueError: 模型未加载时抛出
            Exception: 编码过程中的其他错误（如内存不足、文本格式错误等）
            
        Note:
            - 向量已进行L2标准化，可直接用于余弦相似度计算
            - 批量处理比单个文本编码更高效
            - 空字符串会被编码为零向量
        
        Example:
            >>> texts = ["这是第一个文档", "这是第二个文档"]
            >>> vectors = model.encode_texts(texts)
            >>> print(vectors.shape)  # (2, 512) 假设模型维度为512
        """
        if self.model is None:
            raise ValueError("模型未加载，请先调用 load_model()")
        
        try:
            embeddings = self.model.encode(texts, normalize_embeddings=True)
            return embeddings
        except Exception as e:
            logger.error(f"文本编码失败: {str(e)}")
            raise
    
    def get_embedding_dim(self) -> int:
        """
        获取嵌入向量维度
        
        Returns:
            int: 嵌入向量维度，如果模型未加载则返回None
        """
        return self.embedding_dim
    
    def is_loaded(self) -> bool:
        """
        检查模型是否已加载
        
        Returns:
            bool: 模型已加载返回True，否则返回False
        """
        return self.model is not None