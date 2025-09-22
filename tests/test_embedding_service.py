#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
嵌入和检索模块测试脚本
"""

import sys
import os
from pathlib import Path

# 添加src目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.document_processor import DocumentProcessor
from src.embedding_model import EmbeddingModel
from src.vector_store import VectorStore
from src.retriever import RAGRetriever

def test_embedding_service():
    """测试嵌入服务"""
    print("=== 嵌入服务测试 ===\n")
    
    try:
        # 初始化嵌入服务
        embedding_service = EmbeddingService()
        
        # 测试单个文本向量化
        test_text = "这是一个测试文本，用于验证嵌入功能。"
        print(f"📝 测试文本: {test_text}")
        
        embedding = embedding_service.encode_texts([test_text])
        print(f"✅ 单文本向量化成功，维度: {embedding.shape}")
        
        # 测试批量文本向量化
        test_texts = [
            "RAG是一种结合检索和生成的AI技术。",
            "文档分块是RAG系统的重要步骤。",
            "向量化可以将文本转换为数值表示。",
            "FAISS是一个高效的相似度检索库。"
        ]
        
        print(f"\n📚 测试批量向量化，文本数量: {len(test_texts)}")
        embeddings = embedding_service.encode_texts(test_texts)
        print(f"✅ 批量向量化成功，形状: {embeddings.shape}")
        
        return embedding_service
        
    except Exception as e:
        print(f"❌ 嵌入服务测试失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def test_vector_store(embedding_service):
    """测试向量存储"""
    print("\n=== 向量存储测试 ===\n")
    
    try:
        # 初始化文档处理器
        processor = DocumentProcessor(chunk_size=300, chunk_overlap=50)
        
        # 处理测试文档
        test_doc_path = "data/documents/test_document.md"
        if not os.path.exists(test_doc_path):
            print(f"❌ 测试文档不存在: {test_doc_path}")
            return None
        
        chunks = processor.process_document(test_doc_path)
        print(f"📄 加载文档分块: {len(chunks)} 个")
        
        # 初始化向量存储
        vector_store = VectorStore(embedding_service)
        
        # 构建索引
        print("🔄 构建向量索引...")
        vector_store.build_index(chunks)
        print("✅ 索引构建成功")
        
        # 获取统计信息
        stats = vector_store.get_stats()
        print(f"📊 向量存储统计:")
        print(f"  - 总分块数: {stats['total_chunks']}")
        print(f"  - 索引大小: {stats['index_size']}")
        print(f"  - 嵌入维度: {stats['embedding_dim']}")
        
        return vector_store
            
    except Exception as e:
        print(f"❌ 向量存储测试失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def test_retrieval(vector_store):
    """测试检索功能"""
    print("\n=== 检索功能测试 ===\n")
    
    try:
        # 测试查询
        test_queries = [
            "什么是RAG技术？",
            "如何进行文档分块？",
            "BGE模型的特点",
            "向量化的作用"
        ]
        
        for i, query in enumerate(test_queries, 1):
            print(f"🔍 查询 {i}: {query}")
            
            results = vector_store.search(query, top_k=3)
            
            if results:
                print(f"✅ 找到 {len(results)} 个相关结果:")
                for j, result in enumerate(results, 1):
                    print(f"  {j}. 相似度: {result.score:.3f}")
                    print(f"     内容: {result.content[:100]}...")
                    print(f"     来源: {result.metadata.get('source', '未知')}")
                print()
            else:
                print("❌ 未找到相关结果\n")
        
        return True
        
    except Exception as e:
        print(f"❌ 检索测试失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_index_persistence(vector_store):
    """测试索引持久化"""
    print("=== 索引持久化测试 ===\n")
    
    try:
        # 保存索引
        print("💾 保存索引...")
        vector_store.save_index("test_index")
        print("✅ 索引保存成功")
        
        # 创建新的向量存储实例
        new_embedding_service = EmbeddingService()
        new_vector_store = VectorStore(new_embedding_service)
        
        # 加载索引
        print("📂 加载索引...")
        new_vector_store.load_index("test_index")
        print("✅ 索引加载成功")
        
        # 测试加载后的检索功能
        test_query = "RAG技术的应用"
        results = new_vector_store.search(test_query, top_k=2)
        
        if results:
            print(f"✅ 加载后检索正常，找到 {len(results)} 个结果")
            return True
        else:
            print("❌ 加载后检索失败")
            return False
            
    except Exception as e:
        print(f"❌ 索引持久化测试失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主测试函数"""
    print("🚀 开始嵌入和检索模块测试\n")
    
    # 测试嵌入服务
    embedding_service = test_embedding_service()
    if not embedding_service:
        return
    
    # 测试向量存储
    vector_store = test_vector_store(embedding_service)
    if not vector_store:
        return
    
    # 测试检索功能
    retrieval_success = test_retrieval(vector_store)
    if not retrieval_success:
        return
    
    # 测试索引持久化
    persistence_success = test_index_persistence(vector_store)
    
    if persistence_success:
        print("🎉 所有测试通过！嵌入和检索模块功能正常。")
    else:
        print("⚠️  部分测试失败，请检查日志。")

if __name__ == "__main__":
    main()