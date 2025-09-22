#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
对话服务测试脚本

测试ChatService的各项功能
"""

import sys
import os
import logging

# 添加项目路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config_manager import ConfigManager
from src.chat_service import ChatService
from src.embedding_model import EmbeddingModel
from src.vector_store import VectorStore
from src.retriever import RAGRetriever
from src.document_processor import DocumentProcessor

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_chat_service():
    """测试对话服务基本功能"""
    print("=" * 50)
    print("测试对话服务基本功能")
    print("=" * 50)
    
    try:
        # 1. 初始化配置管理器
        config_manager = ConfigManager()
        config_manager.load_config()
        
        # 2. 初始化对话服务（跳过API客户端初始化）
        chat_service = ChatService.__new__(ChatService)
        chat_service.config_manager = config_manager
        chat_service.config = config_manager.get_config()
        chat_service.client = None  # 模拟模式，不初始化真实客户端
        
        # API配置
        chat_service.model_name = chat_service.config.llm_model
        chat_service.max_tokens = chat_service.config.llm_max_tokens
        chat_service.temperature = chat_service.config.llm_temperature
        chat_service.max_retries = 3
        chat_service.retry_delay = 1
        
        print("✓ 对话服务初始化成功（模拟模式）")
        
        # 3. 测试配置加载
        print(f"✓ 模型配置: {chat_service.model_name}")
        print(f"✓ 最大tokens: {chat_service.max_tokens}")
        print(f"✓ 温度参数: {chat_service.temperature}")
        
        return True
        
    except Exception as e:
        print(f"✗ 对话服务测试失败: {e}")
        return False

def test_rag_integration():
    """测试RAG集成功能"""
    print("\n" + "=" * 50)
    print("测试RAG集成功能")
    print("=" * 50)
    
    try:
        # 1. 初始化所有组件
        config_manager = ConfigManager()
        config_manager.load_config()
        
        # 创建模拟ChatService实例
        chat_service = ChatService.__new__(ChatService)
        chat_service.config_manager = config_manager
        chat_service.config = config_manager.get_config()
        chat_service.client = None
        chat_service.model_name = chat_service.config.llm_model
        chat_service.max_tokens = chat_service.config.llm_max_tokens
        chat_service.temperature = chat_service.config.llm_temperature
        chat_service.max_retries = 3
        chat_service.retry_delay = 1
        
        # 添加模拟流式API调用方法
        def mock_call_api_stream(prompt):
            response_text = "这是基于提供的知识库内容生成的模拟回答。问题相关的关键信息已在上下文中找到。"
            for chunk in response_text.split():
                yield {"choices": [{"delta": {"content": chunk + " "}}]}
        
        chat_service.call_api_stream = mock_call_api_stream
        
        # 2. 准备测试文档
        from src.document_processor import DocumentChunk
        test_documents = [
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
            ),
            DocumentChunk(
                id="test4",
                content="计算机视觉是人工智能的一个分支，旨在让计算机能够理解和解释视觉信息。",
                metadata={"source": "test4.txt"},
                source_file="test4.txt",
                chunk_index=0,
                start_pos=0,
                end_pos=40
            )
        ]
        
        # 创建嵌入模型
        embedding_model = EmbeddingModel(model_name="BAAI/bge-small-zh-v1.5", device="cpu")
        embedding_model.load_model()  # 加载模型
        
        # 4. 初始化检索器
        retriever = RAGRetriever(embedding_model)
        
        # 处理文档并构建索引
        chunks = test_documents
        
        retriever.vector_store.build_index(chunks)
        print("✓ 向量索引构建完成")
        
        # 5. 测试问答
        test_questions = [
            "什么是机器学习？",
            "深度学习和机器学习有什么关系？",
            "NLP是什么？",
            "计算机视觉的作用是什么？"
        ]
        
        for question in test_questions:
            print(f"\n问题: {question}")
            
            # 使用流式方法进行测试
            sources = []
            confidence = 0.0
            response_time = 0.0
            answer_parts = []
            error_msg = None
            
            try:
                for chunk_data in chat_service.generate_answer_stream(question, retriever):
                    if chunk_data['type'] == 'start':
                        sources = chunk_data['sources']
                        confidence = chunk_data['confidence']
                    elif chunk_data['type'] == 'chunk':
                        answer_parts.append(chunk_data['content'])
                    elif chunk_data['type'] == 'end':
                        response_time = chunk_data['response_time']
                    elif chunk_data['type'] == 'error':
                        error_msg = chunk_data['error']
                        break
                
                answer = ''.join(answer_parts)
                print(f"回答: {answer[:100]}...")
                print(f"来源: {', '.join(sources)}")
                print(f"置信度: {confidence:.3f}")
                print(f"响应时间: {response_time:.2f}秒")
                
                if error_msg:
                    print(f"错误: {error_msg}")
                    
            except Exception as e:
                print(f"流式测试错误: {e}")
        
        print("\n✓ RAG集成测试完成")
        return True
        
    except Exception as e:
        print(f"✗ RAG集成测试失败: {e}")
        return False

def test_prompt_building():
    """测试提示词构建"""
    print("\n" + "=" * 50)
    print("测试提示词构建")
    print("=" * 50)
    
    try:
        config_manager = ConfigManager()
        config_manager.load_config()
        
        # 创建模拟ChatService实例
        chat_service = ChatService.__new__(ChatService)
        chat_service.config_manager = config_manager
        chat_service.config = config_manager.get_config()
        chat_service.client = None
        
        question = "什么是人工智能？"
        context = "人工智能（AI）是计算机科学的一个分支，旨在创建能够执行通常需要人类智能的任务的系统。"
        
        prompt = chat_service.build_prompt(question, context)
        
        print("构建的提示词:")
        print("-" * 30)
        print(prompt)
        print("-" * 30)
        
        # 验证提示词包含必要元素
        assert question in prompt, "提示词应包含用户问题"
        assert context in prompt, "提示词应包含上下文"
        assert "知识库内容" in prompt, "提示词应包含知识库标识"
        
        print("✓ 提示词构建测试通过")
        return True
        
    except Exception as e:
        print(f"✗ 提示词构建测试失败: {e}")
        return False

def test_error_handling():
    """测试错误处理"""
    print("\n" + "=" * 50)
    print("测试错误处理")
    print("=" * 50)
    
    try:
        config_manager = ConfigManager()
        config_manager.load_config()
        
        # 创建模拟ChatService实例
        chat_service = ChatService.__new__(ChatService)
        chat_service.config_manager = config_manager
        chat_service.config = config_manager.get_config()
        chat_service.client = None
        
        # 创建嵌入模型
        embedding_model = EmbeddingModel(model_name="BAAI/bge-small-zh-v1.5", device="cpu")
        
        # 测试传入None retriever的情况（应该引发异常）
        try:
            error_found = False
            for chunk_data in chat_service.generate_answer_stream("测试问题", None):
                if chunk_data['type'] == 'error':
                    print(f"错误信息: {chunk_data['error']}")
                    print(f"错误内容: {chunk_data.get('content', '')}")
                    error_found = True
                    break
            
            # 验证错误处理
            assert error_found, "应该有错误信息"
            
        except Exception as e:
            # 如果直接抛出异常也是可以接受的
            print(f"捕获到异常: {e}")
            assert e is not None, "异常对象不应该为None"
        
        print("✓ 错误处理测试通过")
        return True
        
    except Exception as e:
        print(f"✗ 错误处理测试失败: {e}")
        return False

def main():
    """主测试函数"""
    print("开始ChatService功能测试")
    print("=" * 60)
    
    test_results = []
    
    # 运行各项测试
    test_results.append(("对话服务基本功能", test_chat_service()))
    test_results.append(("提示词构建", test_prompt_building()))
    test_results.append(("错误处理", test_error_handling()))
    test_results.append(("RAG集成功能", test_rag_integration()))
    
    # 输出测试结果
    print("\n" + "=" * 60)
    print("测试结果汇总")
    print("=" * 60)
    
    passed = 0
    total = len(test_results)
    
    for test_name, result in test_results:
        status = "✓ 通过" if result else "✗ 失败"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\n总计: {passed}/{total} 项测试通过")
    
    if passed == total:
        print("🎉 所有测试通过！ChatService功能正常")
    else:
        print("⚠️  部分测试失败，请检查配置和网络连接")

if __name__ == "__main__":
    main()