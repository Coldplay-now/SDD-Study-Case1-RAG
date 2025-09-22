#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
嵌入和检索模块
负责文档向量化和FAISS检索
"""

import os
import pickle
import logging
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
from sentence_transformers import SentenceTransformer
import faiss
from tqdm import tqdm

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

class EmbeddingService:
    """
    嵌入服务类
    
    负责文本向量化处理，使用SentenceTransformer模型将文本转换为高维向量表示。
    支持批量处理和多种预训练模型，主要用于RAG系统中的文档和查询向量化。
    
    主要功能：
    - 加载和管理预训练的嵌入模型
    - 将文本转换为标准化的向量表示
    - 支持批量文本处理以提高效率
    - 自动处理模型加载和设备管理
    
    使用示例：
        >>> service = EmbeddingService("BAAI/bge-small-zh-v1.5", "cpu")
        >>> service.load_model()
        >>> vectors = service.encode_texts(["文本1", "文本2"])
    """
    
    def __init__(self, model_name: str = "BAAI/bge-small-zh-v1.5", device: str = "cpu"):
        """
        初始化嵌入服务
        
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
        logger.info(f"嵌入服务初始化 - 模型: {model_name}, 设备: {device}")
    
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
            >>> vectors = service.encode_texts(texts)
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
        >>> store = VectorStore(embedding_service)
        >>> store.build_index(chunks)
        >>> results = store.search(query, top_k=5)
    """
    
    def __init__(self, embedding_service: EmbeddingService):
        """
        初始化向量存储
        
        Args:
            embedding_service (EmbeddingService): 嵌入服务实例，用于文本向量化
                                                 必须已完成模型加载
        
        Attributes:
            embedding_service (EmbeddingService): 嵌入服务实例
            index (faiss.IndexFlatIP): FAISS索引实例，初始为None
            chunk_metadata (Dict): 存储分块元数据的字典，键为chunk_id
            chunk_count (int): 已存储的分块数量
        """
        self.embedding_service = embedding_service
        self.index = None
        self.chunk_metadata = {}
        self.chunk_count = 0
        logger.info("向量存储初始化完成")
    
    def build_index(self, chunks: List[Any]):
        """
        构建向量索引
        
        将文档分块转换为向量并构建FAISS索引，用于后续的相似度搜索。
        该方法会处理所有输入的文档分块，生成对应的向量表示，并建立高效的搜索索引。
        
        处理流程：
        1. 验证输入分块的有效性
        2. 提取所有分块的文本内容
        3. 批量向量化所有文本
        4. 创建FAISS IndexFlatIP索引
        5. 将向量添加到索引中
        6. 存储分块元数据以便检索时使用
        
        Args:
            chunks (List[Any]): 文档分块列表，每个分块包含：
                              - content: 文本内容
                              - metadata: 额外的元数据信息
        
        Raises:
            ValueError: 输入分块列表为空时抛出
            Exception: 向量化失败或索引构建错误
            
        Note:
            - 索引构建完成后会覆盖之前的索引
            - 建议在构建前确保有足够的内存空间
            - 大量分块可能需要较长的处理时间
            - 向量会自动转换为float32格式以优化性能
            
        Performance:
            - 时间复杂度: O(n * d) 其中n为分块数量，d为向量维度
            - 空间复杂度: O(n * d) 用于存储向量索引
        """
        try:
            logger.info(f"开始构建索引，分块数量: {len(chunks)}")
            
            # 提取文本内容
            texts = [chunk.content for chunk in chunks]
            
            # 生成向量
            embeddings = self.embedding_service.encode_texts(texts)
            
            # 创建FAISS索引
            dimension = embeddings.shape[1]
            self.index = faiss.IndexFlatIP(dimension)  # 使用内积相似度
            
            # 添加向量到索引
            self.index.add(embeddings.astype('float32'))
            
            # 存储元数据
            self.chunk_metadata = {}
            for i, chunk in enumerate(chunks):
                chunk_id = f"chunk_{i}"
                self.chunk_metadata[chunk_id] = {
                    'content': chunk.content,
                    'metadata': chunk.metadata,
                    'index': i
                }
            
            self.chunk_count = len(chunks)
            logger.info(f"索引构建完成，向量数量: {self.index.ntotal}")
            
        except Exception as e:
            logger.error(f"索引构建失败: {str(e)}")
            raise
    
    def search(self, query: str, top_k: int = 5, similarity_threshold: float = 0.3) -> List[QueryResult]:
        """
        检索相关文档
        
        基于查询文本在向量索引中搜索最相似的文档分块。使用余弦相似度（通过内积实现）
        来衡量查询与文档的相关性，返回按相似度排序的结果列表。支持相似度阈值过滤，
        确保返回的结果具有足够的相关性。
        
        搜索流程：
        1. 将查询文本向量化
        2. 在FAISS索引中执行相似度搜索
        3. 获取top_k个最相似的结果
        4. 应用相似度阈值过滤低质量结果
        5. 根据索引获取对应的文档内容和元数据
        6. 构建并返回格式化的查询结果
        
        Args:
            query (str): 查询文本，支持中英文混合查询
                        建议使用完整的问题或关键词组合以获得更好的检索效果
            top_k (int, optional): 返回的最相似结果数量，默认为5
                                  范围建议在1-20之间，过大可能影响性能
            similarity_threshold (float, optional): 相似度阈值，默认为0.3
                                                   低于此值的结果将被过滤掉
                                                   范围通常在0.0-1.0之间
            
        Returns:
            List[QueryResult]: 按相似度降序排列的查询结果列表，每个结果包含：
                             - chunk_id: 分块唯一标识符
                             - content: 匹配的文档内容
                             - score: 相似度分数（越高越相似）
                             - metadata: 文档元数据信息
                             - status: 结果状态（"success" 或 "error"）
                             - error_msg: 错误信息（如有）
        
        Raises:
            RuntimeError: 索引未构建时抛出
            Exception: 向量化失败或搜索过程中的其他错误
            
        Note:
            - 搜索结果按相似度分数降序排列
            - 分数范围通常在0-1之间，越接近1表示越相似
            - 相似度阈值可以有效过滤不相关的结果
            - 空查询或过短查询可能返回不准确的结果
            - 搜索性能与索引大小和top_k值相关
            - 如果所有结果都被阈值过滤，会记录警告信息
            
        Performance:
            - 时间复杂度: O(d * n) 其中d为向量维度，n为索引大小
            - 对于大型索引，建议使用适当的top_k值以平衡精度和性能
            
        Example:
            >>> results = store.search("什么是机器学习？", top_k=3, similarity_threshold=0.5)
            >>> for result in results:
            ...     print(f"分数: {result.score:.3f}, 内容: {result.content[:50]}...")
        """
        if self.index is None:
            raise RuntimeError("索引未构建")
        
        try:
            # 对查询进行向量化
            query_embedding = self.embedding_service.encode_texts([query])
            
            # 执行检索
            scores, indices = self.index.search(query_embedding.astype('float32'), top_k)
            
            # 构建结果并应用相似度阈值过滤
            results = []
            filtered_count = 0
            
            for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
                if idx == -1:  # FAISS返回-1表示无效结果
                    continue
                
                # 应用相似度阈值过滤
                if score < similarity_threshold:
                    filtered_count += 1
                    logger.debug(f"过滤低相似度结果: score={score:.3f} < threshold={similarity_threshold}")
                    continue
                
                chunk_id = f"chunk_{idx}"
                if chunk_id in self.chunk_metadata:
                    metadata = self.chunk_metadata[chunk_id]
                    result = QueryResult(
                        chunk_id=chunk_id,
                        content=metadata['content'],
                        score=float(score),
                        metadata=metadata['metadata']
                    )
                    results.append(result)
            
            logger.info(f"检索完成，返回 {len(results)} 个结果，过滤了 {filtered_count} 个低相似度结果")
            
            # 如果没有符合阈值的结果，记录警告
            if len(results) == 0 and filtered_count > 0:
                logger.warning(f"所有检索结果都低于相似度阈值 {similarity_threshold}，可能需要降低阈值或检查查询内容")
            
            return results
            
        except Exception as e:
            logger.error(f"检索失败: {str(e)}")
            return [QueryResult(
                chunk_id="error",
                content="",
                score=0.0,
                metadata={},
                status="error",
                error_msg=str(e)
            )]
    
    def save_index(self, index_path: str):
        """
        保存索引到文件
        
        Args:
            index_path: 索引文件路径
        """
        try:
            if self.index is None:
                raise RuntimeError("索引未构建")
            
            # 创建目录（如果路径包含目录）
            dir_path = os.path.dirname(index_path)
            if dir_path:
                os.makedirs(dir_path, exist_ok=True)
            
            # 保存FAISS索引
            faiss.write_index(self.index, f"{index_path}.faiss")
            
            # 保存元数据
            with open(f"{index_path}.metadata", 'wb') as f:
                pickle.dump({
                    'chunk_metadata': self.chunk_metadata,
                    'chunk_count': self.chunk_count
                }, f)
            
            logger.info(f"索引保存成功: {index_path}")
            
        except Exception as e:
            logger.error(f"索引保存失败: {str(e)}")
            raise
    
    def load_index(self, index_path: str):
        """
        从文件加载索引
        
        Args:
            index_path: 索引文件路径
        """
        try:
            # 加载FAISS索引
            self.index = faiss.read_index(f"{index_path}.faiss")
            
            # 加载元数据
            with open(f"{index_path}.metadata", 'rb') as f:
                data = pickle.load(f)
                self.chunk_metadata = data['chunk_metadata']
                self.chunk_count = data['chunk_count']
            
            logger.info(f"索引加载成功: {index_path}, 向量数量: {self.index.ntotal}")
            
        except Exception as e:
            logger.error(f"索引加载失败: {str(e)}")
            raise
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            'total_chunks': self.chunk_count,
            'index_size': self.index.ntotal if self.index else 0,
            'embedding_dim': self.embedding_service.embedding_dim
        }

class RAGRetriever:
    """
    RAG检索器
    
    高级的检索增强生成（RAG）检索器，整合了嵌入服务和向量存储功能。
    提供端到端的文档检索解决方案，从文档处理到相似度搜索的完整流程。
    
    主要功能：
    - 自动管理嵌入服务和向量存储
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
    
    def __init__(self, embedding_service: EmbeddingService = None):
        """
        初始化RAG检索器
        
        Args:
            embedding_service (EmbeddingService, optional): 嵌入服务实例
                                                          如果为None，将使用默认配置创建新实例
        
        Attributes:
            embedding_service (EmbeddingService): 嵌入服务实例
            vector_store (VectorStore): 向量存储实例
        
        Note:
            - 如果未提供embedding_service，会自动创建默认实例
            - 默认使用BAAI/bge-small-zh-v1.5模型和CPU设备
            - 创建后需要调用build_index_from_documents构建索引
        """
        self.embedding_service = embedding_service or EmbeddingService()
        self.vector_store = VectorStore(self.embedding_service)
        self.document_processor = None
        logger.info("RAG检索器初始化完成")
    
    def build_index_from_documents(self, documents: List[str]):
        """
        从文档列表构建索引
        
        Args:
            documents: 文档路径列表
        """
        try:
            from document_processor import DocumentProcessor
            
            if self.document_processor is None:
                self.document_processor = DocumentProcessor()
            
            all_chunks = []
            for doc_path in documents:
                logger.info(f"处理文档: {doc_path}")
                chunks = self.document_processor.process_document(doc_path)
                all_chunks.extend(chunks)
            
            logger.info(f"总共处理了 {len(all_chunks)} 个分块")
            
            # 构建向量索引
            self.vector_store.build_index(all_chunks)
            
        except Exception as e:
            logger.error(f"从文档构建索引失败: {str(e)}")
            raise
    
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