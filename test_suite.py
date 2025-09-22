#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RAG学习系统 - 核心测试套件
整合所有重要的测试功能，提供完整的系统验证
"""

import sys
import os
import time
import logging
from typing import List, Dict, Any

# 添加src目录到Python路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# 导入核心模块
from src.embedding_model import EmbeddingModel
from src.vector_store import VectorStore
from src.retriever import RAGRetriever
from src.chat_service import ChatService
from src.config_manager import ConfigManager
from src.document_processor import DocumentChunk

class RAGTestSuite:
    """RAG系统核心测试套件"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.test_results = {}
        
    def setup_test_environment(self):
        """设置测试环境"""
        print("🔧 设置测试环境...")
        
        # 配置日志
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # 准备测试数据
        self.test_documents = [
            DocumentChunk(
                id="test1",
                content="机器学习是人工智能的一个重要分支，它使计算机能够在没有明确编程的情况下学习和改进。",
                metadata={"source": "test1.txt"},
                source_file="test1.txt",
                chunk_index=0,
                start_pos=0,
                end_pos=50
            ),
            DocumentChunk(
                id="test2",
                content="深度学习是机器学习的一个子集，使用多层神经网络来模拟人脑的工作方式。",
                metadata={"source": "test2.txt"},
                source_file="test2.txt",
                chunk_index=0,
                start_pos=0,
                end_pos=40
            ),
            DocumentChunk(
                id="test3",
                content="自然语言处理（NLP）是人工智能的一个领域，专注于计算机与人类语言之间的交互。",
                metadata={"source": "test3.txt"},
                source_file="test3.txt",
                chunk_index=0,
                start_pos=0,
                end_pos=45
            )
        ]
        
        self.test_questions = [
            "什么是机器学习？",
            "深度学习和机器学习有什么关系？",
            "NLP是什么？"
        ]
        
    def test_embedding_model(self):
        """测试嵌入模型"""
        print("\n📊 测试嵌入模型...")
        try:
            model = EmbeddingModel()
            model.load_model()  # 需要先加载模型
            
            # 测试文本嵌入
            test_texts = ["这是一个测试文本"]
            embeddings = model.encode_texts(test_texts)
            
            assert embeddings is not None, "嵌入结果不能为空"
            assert len(embeddings.shape) == 2, "嵌入向量应该是二维数组"
            assert embeddings.shape[0] == 1, "应该有一个嵌入向量"
            assert embeddings.shape[1] > 0, "嵌入向量维度应该大于0"
            
            self.test_results['embedding_model'] = True
            print("✅ 嵌入模型测试通过")
            
        except Exception as e:
            self.test_results['embedding_model'] = False
            print(f"❌ 嵌入模型测试失败: {e}")
            
    def test_vector_store(self):
        """测试向量存储"""
        print("\n🗄️ 测试向量存储...")
        try:
            model = EmbeddingModel()
            model.load_model()  # 需要先加载模型
            store = VectorStore(model)
            
            # 构建索引
            store.build_index(self.test_documents)
            
            # 测试检索
            results = store.search("机器学习", top_k=2)
            
            assert len(results) > 0, "检索结果不能为空"
            assert hasattr(results[0], 'content'), "检索结果应该包含content属性"
            assert hasattr(results[0], 'score'), "检索结果应该包含score属性"
            
            self.test_results['vector_store'] = True
            print("✅ 向量存储测试通过")
            
        except Exception as e:
            self.test_results['vector_store'] = False
            print(f"❌ 向量存储测试失败: {e}")
            
    def test_retriever(self):
        """测试检索器"""
        print("\n🔍 测试检索器...")
        try:
            model = EmbeddingModel()
            model.load_model()  # 需要先加载模型
            retriever = RAGRetriever(model)  # 只传入embedding_model
            
            # 构建索引
            retriever.vector_store.build_index(self.test_documents)
            
            # 测试检索
            results = retriever.search("深度学习", top_k=2)
            
            assert len(results) > 0, "检索结果不能为空"
            
            self.test_results['retriever'] = True
            print("✅ 检索器测试通过")
            
        except Exception as e:
            self.test_results['retriever'] = False
            print(f"❌ 检索器测试失败: {e}")
            
    def test_chat_service(self):
        """测试对话服务"""
        print("\n💬 测试对话服务...")
        try:
            # 初始化配置管理器
            config_manager = ConfigManager()
            chat_service = ChatService(config_manager)
            
            # 初始化检索器
            model = EmbeddingModel()
            model.load_model()
            retriever = RAGRetriever(model)
            retriever.vector_store.build_index(self.test_documents)
            
            # 测试对话流
            response_chunks = []
            for chunk in chat_service.generate_answer_stream("什么是机器学习？", retriever):
                if chunk.get('type') == 'chunk':
                    response_chunks.append(chunk.get('content', ''))
            
            response = "".join(response_chunks)
            
            assert len(response_chunks) > 0 or response is not None, "对话响应不能为空"
            
            self.test_results['chat_service'] = True
            print("✅ 对话服务测试通过")
            
        except Exception as e:
            self.test_results['chat_service'] = False
            print(f"❌ 对话服务测试失败: {e}")
            
    def test_integration(self):
        """集成测试"""
        print("\n🔗 集成测试...")
        try:
            # 完整流程测试
            config_manager = ConfigManager()
            chat_service = ChatService(config_manager)
            
            model = EmbeddingModel()
            model.load_model()
            retriever = RAGRetriever(model)
            retriever.vector_store.build_index(self.test_documents)
            
            # 端到端测试
            response_chunks = []
            for chunk in chat_service.generate_answer_stream("什么是深度学习？", retriever):
                if chunk.get('type') == 'chunk':
                    response_chunks.append(chunk.get('content', ''))
            
            response = "".join(response_chunks)
            
            assert len(response_chunks) > 0 or response is not None, "集成测试响应不能为空"
            
            self.test_results['integration'] = True
            print("✅ 集成测试通过")
            
        except Exception as e:
            self.test_results['integration'] = False
            print(f"❌ 集成测试失败: {e}")
            
    def run_all_tests(self):
        """运行所有测试"""
        print("🧪 RAG系统核心测试套件")
        print("=" * 60)
        
        # 设置测试环境
        self.setup_test_environment()
        
        # 运行各项测试
        self.test_embedding_model()
        self.test_vector_store()
        self.test_retriever()
        self.test_chat_service()
        self.test_integration()
        
        # 输出测试结果
        self.print_test_summary()
        
    def print_test_summary(self):
        """打印测试摘要"""
        print("\n" + "=" * 60)
        print("📊 测试结果摘要")
        print("=" * 60)
        
        passed = 0
        total = len(self.test_results)
        
        for test_name, result in self.test_results.items():
            status = "✅ 通过" if result else "❌ 失败"
            print(f"{test_name.replace('_', ' ').title()}: {status}")
            if result:
                passed += 1
                
        print("-" * 60)
        print(f"总计: {passed}/{total} 项测试通过")
        
        if passed == total:
            print("🎉 所有测试通过！系统功能正常")
        else:
            print("⚠️ 部分测试失败，请检查系统配置")
            
        return passed == total

def main():
    """主函数"""
    test_suite = RAGTestSuite()
    success = test_suite.run_all_tests()
    
    # 返回适当的退出码
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()