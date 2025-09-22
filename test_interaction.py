#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RAG学习系统 - 交互测试脚本
测试系统的完整交互功能，包括多轮对话、特殊命令等
"""

import os
import sys
import time
import logging
from typing import List, Dict, Any

# 添加src目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

try:
    from src.config_manager import ConfigManager
    from src.document_processor import DocumentProcessor
    from src.embedding_model import EmbeddingModel
    from src.vector_store import VectorStore
    from src.retriever import RAGRetriever
    from src.chat_service import ChatService
except ImportError:
    # 如果相对导入失败，尝试直接导入
    import config_manager
    import document_processor
    import embedding_model
    import vector_store
    import retriever
    import chat_service
    
    ConfigManager = config_manager.ConfigManager
    DocumentProcessor = document_processor.DocumentProcessor
    EmbeddingModel = embedding_model.EmbeddingModel
    VectorStore = vector_store.VectorStore
    RAGRetriever = retriever.RAGRetriever
    ChatService = chat_service.ChatService

def setup_logging():
    """设置日志"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('test_interaction.log', encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def test_system_initialization():
    """测试系统初始化"""
    logger = logging.getLogger(__name__)
    logger.info("🔧 开始系统初始化测试...")
    
    try:
        # 初始化配置管理器
        config_manager = ConfigManager()
        config_manager.load_config()  # 加载配置
        
        # 获取配置
        config = config_manager.get_config()
        logger.info("✅ 配置加载成功")
        
        # 初始化文档处理器
        doc_processor = DocumentProcessor()
        logger.info("✅ 文档处理器初始化成功")
        
        # 初始化嵌入模型
        embedding_model = EmbeddingModel(config_manager)
        logger.info("✅ 嵌入模型初始化成功")
        
        # 初始化向量存储
        vector_store = VectorStore(config_manager)
        logger.info("✅ 向量存储初始化成功")
        
        # 初始化RAG检索器
        retriever = RAGRetriever(config_manager)
        logger.info("✅ RAG检索器初始化成功")
        
        # 初始化对话服务
        chat_service = ChatService(config_manager)
        logger.info("✅ 对话服务初始化成功")
        
        logger.info("🎉 系统初始化测试完成")
        
        return {
            'config_manager': config_manager,
            'doc_processor': doc_processor,
            'embedding_model': embedding_model,
            'vector_store': vector_store,
            'retriever': retriever,
            'chat_service': chat_service
        }
        
    except Exception as e:
        logger.error(f"❌ 系统初始化失败: {e}")
        return None

def test_document_processing(components):
    """测试文档处理功能"""
    logger = logging.getLogger(__name__)
    logger.info("📚 开始文档处理测试...")
    
    try:
        doc_processor = components['doc_processor']
        config_manager = components['config_manager']
        config = config_manager.get_config()
        
        # 获取文档目录
        docs_dir = getattr(config.paths, 'documents', './data/documents')
        
        if not os.path.exists(docs_dir):
            logger.warning(f"⚠️  文档目录不存在: {docs_dir}")
            return False
            
        # 处理文档
        all_chunks = []
        for file_name in os.listdir(docs_dir):
            if file_name.endswith('.md'):
                file_path = os.path.join(docs_dir, file_name)
                chunks = doc_processor.process_document(file_path)
                all_chunks.extend(chunks)
                
        logger.info(f"✅ 成功处理 {len(all_chunks)} 个文档块")
        
        if all_chunks:
            # 显示第一个文档块的信息
            first_chunk = all_chunks[0]
            logger.info(f"📄 示例文档: {first_chunk.source_file}")
            logger.info(f"📝 内容长度: {len(first_chunk.content)} 字符")
            
        return True
        
    except Exception as e:
        logger.error(f"❌ 文档处理测试失败: {e}")
        return False

def test_interactive_questions():
    """测试交互式问题"""
    logger = logging.getLogger(__name__)
    logger.info("🧪 开始交互式问题测试...")
    
    # 测试问题列表
    test_questions = [
        "什么是机器学习？",
        "深度学习有什么特点？", 
        "RAG技术是什么？",
        "人工智能的应用领域有哪些？",
        "神经网络的基本原理",
        "自然语言处理的主要任务",
        "计算机视觉技术介绍",
        "强化学习算法原理"
    ]
    
    print("\n" + "=" * 60)
    print("🤖 RAG系统交互测试指南")
    print("=" * 60)
    print()
    print("📋 建议测试的问题列表:")
    print()
    
    for i, question in enumerate(test_questions, 1):
        print(f"  {i:2d}. {question}")
    print()
    
    print("🔧 特殊命令测试:")
    print("  • help  - 查看帮助信息")
    print("  • stats - 查看系统统计")
    print("  • quit  - 退出系统")
    print("  • exit  - 退出系统")
    print()
    
    print("💡 测试建议:")
    print("  1. 先测试基础问题，观察检索和回答质量")
    print("  2. 测试特殊命令功能是否正常")
    print("  3. 尝试多轮对话，测试上下文理解")
    print("  4. 测试边界情况（空输入、超长输入等）")
    print("  5. 观察响应时间和系统性能")
    print()
    
    print("📊 观察要点:")
    print("  • 检索结果的相关性和数量")
    print("  • 回答的准确性和完整性")
    print("  • 相似度阈值过滤效果")
    print("  • 系统响应时间")
    print("  • 错误处理机制")
    print()
    
    print("🎯 质量评估标准:")
    print("  • 回答是否基于检索到的文档内容")
    print("  • 回答是否准确回应了用户问题")
    print("  • 系统是否正确显示了参考来源")
    print("  • 置信度评分是否合理")
    print()
    
    print("=" * 60)
    print("请在运行的RAG系统中逐一测试上述问题")
    print("观察每个问题的检索过程和回答质量")
    print("=" * 60)

def main():
    """主测试函数"""
    logger = setup_logging()
    logger.info("🚀 开始RAG系统交互测试")
    
    # 显示交互测试指南
    test_interactive_questions()
    
    # 可选：进行系统组件测试
    print("\n🔧 是否需要进行系统组件测试？(y/n): ", end="")
    try:
        choice = input().strip().lower()
        if choice == 'y':
            logger.info("开始系统组件测试...")
            components = test_system_initialization()
            if components:
                test_document_processing(components)
                logger.info("✅ 系统组件测试完成")
            else:
                logger.error("❌ 系统组件测试失败")
    except KeyboardInterrupt:
        print("\n测试已取消")
    except Exception as e:
        logger.error(f"测试过程中出错: {e}")

if __name__ == "__main__":
    main()