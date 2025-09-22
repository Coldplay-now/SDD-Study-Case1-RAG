#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
向量存储模块
专门负责FAISS索引管理和向量检索
"""

import os
import pickle
import logging
import numpy as np
from typing import List, Dict, Any
from dataclasses import dataclass
import faiss
from tqdm import tqdm

from .embedding_model import EmbeddingModel

logger = logging.getLogger(__name__)

@dataclass
class QueryResult:
    """查询结果数据结构"""
    chunk_id: str
    content: str
    score: float
    metadata: Dict[str, Any]
    status: str = "success"
    error_msg: str = ""

class VectorStore:
    """
    向量存储类
    
    基于FAISS库实现的高效向量存储和检索系统。支持大规模向量的快速相似度搜索，
    使用IndexFlatIP索引实现精确的内积搜索，适用于RAG系统中的文档检索。
    
    主要功能：
    - 构建和管理FAISS向量索引
    - 高效的向量相似度搜索
    - 索引的持久化存储和加载
    - 向量数据库统计信息查询
    
    技术特点：
    - 使用FAISS IndexFlatIP实现精确搜索
    - 支持批量向量添加和搜索
    - 内存高效的向量存储
    - 可扩展的索引结构
    
    使用示例：
        >>> store = VectorStore(embedding_model)
        >>> store.build_index(chunks)
        >>> results = store.search(query, top_k=5)
    """
    
    def __init__(self, embedding_model: EmbeddingModel):
        """
        初始化向量存储
        
        Args:
            embedding_model (EmbeddingModel): 嵌入模型实例，用于文本向量化
                                             必须已完成模型加载
        
        Attributes:
            embedding_model (EmbeddingModel): 嵌入模型实例
            index (faiss.IndexFlatIP): FAISS索引实例，初始为None
            chunk_metadata (Dict): 存储分块元数据的字典，键为chunk_id
            chunk_count (int): 已存储的分块数量
        """
        self.embedding_model = embedding_model
        self.index = None
        self.chunk_metadata = {}
        self.chunk_count = 0
        logger.info("向量存储初始化完成")
    
    def build_index(self, chunks: List[Any]):
        """
        构建向量索引
        
        从文档分块列表构建FAISS向量索引。处理流程包括文本向量化、索引创建和元数据存储。
        使用IndexFlatIP索引类型，支持精确的内积相似度搜索。
        
        处理步骤：
        1. 提取所有分块的文本内容
        2. 批量向量化所有文本
        3. 创建FAISS索引并添加向量
        4. 存储分块元数据以便检索时使用
        
        Args:
            chunks (List[Any]): 文档分块列表，每个分块应包含以下属性：
                               - content: 文本内容
                               - chunk_id: 唯一标识符
                               - metadata: 元数据字典（可选）
                               - source: 来源信息（可选）
        
        Raises:
            ValueError: 当分块列表为空或格式不正确时
            Exception: 向量化或索引构建过程中的错误
            
        Note:
            - 构建过程可能需要较长时间，取决于分块数量和模型性能
            - 索引构建完成后会自动存储元数据映射
            - 重复调用会覆盖之前的索引
        
        Example:
            >>> chunks = [chunk1, chunk2, chunk3]  # 文档分块列表
            >>> store.build_index(chunks)
            >>> print(f"索引包含 {store.chunk_count} 个向量")
        """
        if not chunks:
            raise ValueError("分块列表不能为空")
        
        try:
            logger.info(f"开始构建向量索引，共 {len(chunks)} 个分块")
            
            # 提取文本内容
            texts = [chunk.content for chunk in chunks]
            
            # 向量化
            logger.info("正在进行文本向量化...")
            embeddings = self.embedding_model.encode_texts(texts)
            
            # 创建FAISS索引
            dimension = embeddings.shape[1]
            self.index = faiss.IndexFlatIP(dimension)
            
            # 添加向量到索引
            logger.info("正在构建FAISS索引...")
            self.index.add(embeddings.astype('float32'))
            
            # 存储元数据
            self.chunk_metadata = {}
            for i, chunk in enumerate(chunks):
                self.chunk_metadata[i] = {
                    'chunk_id': chunk.id,
                    'content': chunk.content,
                    'metadata': getattr(chunk, 'metadata', {}),
                    'source': getattr(chunk, 'source', '')
                }
            
            self.chunk_count = len(chunks)
            logger.info(f"向量索引构建完成，包含 {self.chunk_count} 个向量，维度: {dimension}")
            
        except Exception as e:
            logger.error(f"索引构建失败: {str(e)}")
            raise
    
    def search(self, query: str, top_k: int = 5, similarity_threshold: float = 0.3) -> List[QueryResult]:
        """
        执行向量检索
        
        根据查询文本在向量索引中搜索最相似的文档分块。使用余弦相似度进行排序，
        并根据相似度阈值过滤结果。
        
        检索流程：
        1. 将查询文本向量化
        2. 在FAISS索引中执行相似度搜索
        3. 根据相似度阈值过滤结果
        4. 构造并返回查询结果对象
        
        Args:
            query (str): 查询文本，支持中英文混合
            top_k (int, optional): 返回的最大结果数量，默认为5
            similarity_threshold (float, optional): 相似度阈值，范围[0,1]，默认0.3
                                                   低于此阈值的结果将被过滤
        
        Returns:
            List[QueryResult]: 查询结果列表，按相似度降序排列
                              每个结果包含分块ID、内容、相似度分数和元数据
        
        Raises:
            ValueError: 索引未构建或查询文本为空时
            Exception: 向量化或搜索过程中的错误
            
        Note:
            - 相似度分数范围为[0,1]，1表示完全匹配
            - 结果已按相似度降序排列
            - 空查询会返回空结果列表
        
        Example:
            >>> results = store.search("什么是RAG技术？", top_k=3, similarity_threshold=0.5)
            >>> for result in results:
            ...     print(f"相似度: {result.score:.3f}, 内容: {result.content[:50]}...")
        """
        if self.index is None:
            raise ValueError("索引未构建，请先调用 build_index()")
        
        if not query.strip():
            logger.warning("查询文本为空")
            return []
        
        try:
            # 向量化查询
            query_embedding = self.embedding_model.encode_texts([query])
            
            # 执行搜索
            scores, indices = self.index.search(query_embedding.astype('float32'), top_k)
            
            # 构造结果
            results = []
            for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
                if idx == -1:  # FAISS返回-1表示无效结果
                    continue
                
                if score < similarity_threshold:
                    continue
                
                chunk_info = self.chunk_metadata[idx]
                result = QueryResult(
                    chunk_id=chunk_info['chunk_id'],
                    content=chunk_info['content'],
                    score=float(score),
                    metadata=chunk_info['metadata']
                )
                results.append(result)
            
            logger.info(f"检索完成，返回 {len(results)} 个结果")
            return results
            
        except Exception as e:
            logger.error(f"检索失败: {str(e)}")
            return [QueryResult(
                chunk_id="",
                content="",
                score=0.0,
                metadata={},
                status="error",
                error_msg=str(e)
            )]
    
    def save_index(self, index_path: str):
        """
        保存索引到文件
        
        将FAISS索引和相关元数据持久化到磁盘。索引文件使用.faiss扩展名，
        元数据使用.pkl扩展名。
        
        Args:
            index_path (str): 索引文件路径（不包含扩展名）
                             实际会生成两个文件：
                             - {index_path}.faiss: FAISS索引文件
                             - {index_path}.pkl: 元数据文件
        
        Raises:
            ValueError: 索引未构建时
            Exception: 文件写入错误
            
        Example:
            >>> store.save_index("data/vectors/my_index")
            # 生成文件: data/vectors/my_index.faiss 和 data/vectors/my_index.pkl
        """
        if self.index is None:
            raise ValueError("索引未构建，无法保存")
        
        try:
            # 确保目录存在
            os.makedirs(os.path.dirname(index_path), exist_ok=True)
            
            # 保存FAISS索引
            faiss.write_index(self.index, f"{index_path}.faiss")
            
            # 保存元数据
            metadata = {
                'chunk_metadata': self.chunk_metadata,
                'chunk_count': self.chunk_count
            }
            with open(f"{index_path}.pkl", 'wb') as f:
                pickle.dump(metadata, f)
            
            logger.info(f"索引已保存到: {index_path}")
            
        except Exception as e:
            logger.error(f"索引保存失败: {str(e)}")
            raise
    
    def load_index(self, index_path: str):
        """
        从文件加载索引
        
        从磁盘加载之前保存的FAISS索引和元数据。
        
        Args:
            index_path (str): 索引文件路径（不包含扩展名）
                             需要存在对应的.faiss和.pkl文件
        
        Raises:
            FileNotFoundError: 索引文件不存在时
            Exception: 文件读取或索引加载错误
            
        Example:
            >>> store.load_index("data/vectors/my_index")
            >>> print(f"加载了 {store.chunk_count} 个向量")
        """
        try:
            # 加载FAISS索引
            if not os.path.exists(f"{index_path}.faiss"):
                raise FileNotFoundError(f"索引文件不存在: {index_path}.faiss")
            
            self.index = faiss.read_index(f"{index_path}.faiss")
            
            # 加载元数据
            if not os.path.exists(f"{index_path}.pkl"):
                raise FileNotFoundError(f"元数据文件不存在: {index_path}.pkl")
            
            with open(f"{index_path}.pkl", 'rb') as f:
                metadata = pickle.load(f)
            
            self.chunk_metadata = metadata['chunk_metadata']
            self.chunk_count = metadata['chunk_count']
            
            logger.info(f"索引加载成功，包含 {self.chunk_count} 个向量")
            
        except Exception as e:
            logger.error(f"索引加载失败: {str(e)}")
            raise
    
    def get_stats(self) -> Dict[str, Any]:
        """
        获取向量存储统计信息
        
        Returns:
            Dict[str, Any]: 包含以下统计信息的字典：
                           - total_vectors: 总向量数量
                           - index_size: 索引大小（字节）
                           - dimension: 向量维度
                           - is_trained: 索引是否已训练
        """
        if self.index is None:
            return {
                'total_vectors': 0,
                'index_size': 0,
                'dimension': 0,
                'is_trained': False
            }
        
        return {
            'total_vectors': self.index.ntotal,
            'index_size': self.index.ntotal * self.index.d * 4,  # 假设float32
            'dimension': self.index.d,
            'is_trained': self.index.is_trained
        }