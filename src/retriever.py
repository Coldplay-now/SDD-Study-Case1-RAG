#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RAG检索器模块
专门负责检索增强生成的检索逻辑
"""

import logging
from typing import List, Dict, Any

from .embedding_model import EmbeddingModel
from .vector_store import VectorStore, QueryResult

logger = logging.getLogger(__name__)

class RAGRetriever:
    """
    RAG检索器
    
    高级的检索增强生成（RAG）检索器，整合了嵌入模型和向量存储功能。
    提供端到端的文档检索解决方案，从文档处理到相似度搜索的完整流程。
    
    主要功能：
    - 自动管理嵌入模型和向量存储
    - 从原始文档构建搜索索引
    - 提供统一的检索接口
    - 支持索引的持久化和恢复
    
    设计特点：
    - 封装了底层的向量化和索引操作
    - 提供简化的API接口
    - 自动处理服务初始化和配置
    - 支持灵活的检索参数配置
    
    适用场景：
    - 问答系统的文档检索
    - 知识库搜索
    - 相似文档推荐
    - 内容匹配和过滤
    
    使用示例：
        >>> retriever = RAGRetriever()
        >>> retriever.build_index_from_documents(documents)
        >>> results = retriever.search("查询问题", top_k=5)
    """
    
    def __init__(self, embedding_model: EmbeddingModel = None):
        """
        初始化RAG检索器
        
        Args:
            embedding_model (EmbeddingModel, optional): 嵌入模型实例
                                                       如果为None，将使用默认配置创建新实例
        
        Attributes:
            embedding_model (EmbeddingModel): 嵌入模型实例
            vector_store (VectorStore): 向量存储实例
        
        Note:
            - 如果未提供embedding_model，会自动创建默认实例
            - 默认使用BAAI/bge-small-zh-v1.5模型和CPU设备
            - 创建后需要调用build_index_from_documents构建索引
        """
        self.embedding_model = embedding_model or EmbeddingModel()
        self.vector_store = VectorStore(self.embedding_model)
        self.document_processor = None
        logger.info("RAG检索器初始化完成")
    
    def build_index_from_documents(self, documents: List[str]):
        """
        从文档列表构建索引
        
        Args:
            documents: 文档列表
        """
        # 这里可以添加文档处理逻辑
        # 目前简化处理，假设documents已经是处理好的分块
        logger.info(f"开始从 {len(documents)} 个文档构建索引")
        
        # 创建简单的文档分块对象
        class SimpleChunk:
            def __init__(self, content, chunk_id, source=""):
                self.content = content
                self.chunk_id = chunk_id
                self.source = source
                self.metadata = {"source": source}
        
        chunks = [SimpleChunk(doc, f"chunk_{i}", f"doc_{i}") 
                 for i, doc in enumerate(documents)]
        
        self.vector_store.build_index(chunks)
        logger.info("索引构建完成")
    
    def search(self, query: str, top_k: int = 5, similarity_threshold: float = 0.3) -> List[QueryResult]:
        """
        执行检索
        
        Args:
            query: 查询文本
            top_k: 返回结果数量
            similarity_threshold: 相似度阈值，低于此值的结果将被过滤
            
        Returns:
            查询结果列表
        """
        return self.vector_store.search(query, top_k, similarity_threshold)
    
    def save_index(self, index_path: str):
        """保存索引"""
        self.vector_store.save_index(index_path)
    
    def load_index(self, index_path: str):
        """加载索引"""
        self.vector_store.load_index(index_path)
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return self.vector_store.get_stats()
    
    def is_ready(self) -> bool:
        """
        检查检索器是否准备就绪
        
        Returns:
            bool: 如果嵌入模型已加载且向量存储有索引则返回True
        """
        return (self.embedding_model.is_loaded() and 
                self.vector_store.index is not None)