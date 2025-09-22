#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RAG学习系统 - 主程序入口
版本: v1.0
日期: 2025-01-27
"""

import os
import sys
import time
import logging
from pathlib import Path

# 添加src目录到Python路径
sys.path.append(str(Path(__file__).parent / "src"))

from config_manager import ConfigManager

def setup_logging(log_level: str = "INFO"):
    """
    设置日志配置
    
    配置系统的日志记录功能，包括日志级别、格式和输出目标。
    同时输出到文件和控制台，便于开发调试和生产监控。
    
    Args:
        log_level (str): 日志级别，默认为"INFO"。支持DEBUG、INFO、WARNING、ERROR、CRITICAL
    
    配置内容：
    - 日志级别：可配置的日志级别，记录相应级别及以上的信息
    - 日志格式：包含时间戳、模块名、级别和消息内容
    - 输出目标：同时写入文件(rag_system.log)和控制台
    - 文件编码：UTF-8，支持中文字符
    
    日志文件：
    - 文件名：rag_system.log
    - 位置：当前工作目录
    - 编码：UTF-8
    - 模式：追加写入
    
    Note:
        - 日志文件会在程序运行目录下创建
        - 如果文件已存在，新日志会追加到文件末尾
        - 控制台输出便于实时监控程序运行状态
        - 建议在程序启动时首先调用此函数
        - 可以通过配置文件动态调整日志级别
    """
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)
    logging.basicConfig(
        level=numeric_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('rag_system.log', encoding='utf-8')
        ]
    )

def main():
    """
    主函数 - RAG学习系统的入口点
    
    负责整个RAG系统的初始化、配置和运行。按顺序执行以下步骤：
    1. 系统初始化（日志、配置等）
    2. 核心组件初始化（文档处理器、嵌入服务、向量存储等）
    3. 文档处理和索引构建
    4. 交互式问答服务
    
    系统架构：
    - ConfigManager: 配置管理
    - DocumentProcessor: 文档处理和分块
    - EmbeddingService: 文本向量化
    - VectorStore: 向量存储和检索
    - ChatService: 对话生成服务
    
    运行流程：
    1. 设置日志系统
    2. 初始化各个核心组件
    3. 扫描并处理文档目录中的Markdown文件
    4. 构建向量索引并保存
    5. 启动交互式问答循环
    6. 处理用户输入并生成回答
    
    异常处理：
    - 组件初始化失败时优雅退出
    - 文档处理错误时跳过问题文件
    - 用户交互错误时继续运行
    - 系统级错误时记录日志并退出
    
    用户交互：
    - 支持连续问答
    - 显示回答和置信度
    - 提供参考来源信息
    - 支持优雅退出（quit/exit/退出）
    
    Note:
        - 确保docs目录存在且包含Markdown文件
        - 首次运行需要下载嵌入模型，可能需要较长时间
        - 索引文件会保存到data/vector_index目录
        - 所有操作都会记录到rag_system.log文件
    
    Raises:
        SystemExit: 当关键组件初始化失败时
        KeyboardInterrupt: 用户中断程序时
        Exception: 其他系统级错误
    """
    print("=" * 50)
    print("🚀 RAG学习系统启动中...")
    print("=" * 50)
    
    # 初始化日志（使用默认级别）
    setup_logging()
    logger = logging.getLogger(__name__)
    
    try:
        # 1. 加载配置
        logger.info("开始加载系统配置...")
        config_manager = ConfigManager()
        config = config_manager.load_config()
        
        # 重新设置日志级别（基于配置）
        setup_logging(config.system['log_level'])
        logger = logging.getLogger(__name__)
        
        # 显示配置摘要
        print("📋 配置摘要:")
        print(f"  🤖 嵌入模型: {config.embedding['model_name']}")
        print(f"  🔍 检索设置: top_k={config.retrieval['top_k']}, chunk_size={config.retrieval['chunk_size']}")
        print(f"  💬 LLM模型: {config.llm['model']}")
        print(f"  📊 日志级别: {config.system['log_level']}")
        
        # 2. 初始化各个模块
        logger.info("开始初始化系统模块...")
        
        # 导入模块
        from src.embedding_model import EmbeddingModel
        from src.retriever import RAGRetriever
        from src.document_processor import MarkdownDocumentProcessor
        from src.chat_service import ChatService
        
        # 初始化嵌入模型
        print("🔧 初始化嵌入模型...")
        embedding_model = EmbeddingModel(
            model_name=config.embedding['model_name'],
            device=config.embedding['device']
        )
        embedding_model.load_model()
        
        # 初始化文档处理器
        print("📄 初始化文档处理器...")
        doc_processor = MarkdownDocumentProcessor(
            chunk_size=config.retrieval['chunk_size'],
            chunk_overlap=config.retrieval['chunk_overlap']
        )
        
        # 初始化RAG检索器
        print("🔍 初始化RAG检索器...")
        retriever = RAGRetriever(embedding_model=embedding_model)
        
        # 检查是否有文档需要处理
        documents_dir = Path("data/documents")
        if documents_dir.exists() and any(documents_dir.glob("*.md")):
            print("📚 发现文档，开始处理和索引构建...")
            
            # 处理所有markdown文档
            all_chunks = []
            for doc_file in documents_dir.glob("*.md"):
                logger.info(f"处理文档: {doc_file}")
                chunks = doc_processor.process_document(str(doc_file))
                all_chunks.extend(chunks)
                print(f"  ✓ {doc_file.name}: {len(chunks)} 个分块")
            
            # 构建索引
            if all_chunks:
                print(f"🏗️  构建向量索引 ({len(all_chunks)} 个分块)...")
                retriever.vector_store.build_index(all_chunks)
                
                # 保存索引
                index_path = "data/vectors/main_index"
                retriever.save_index(index_path)
                print(f"💾 索引已保存到: {index_path}")
            else:
                print("⚠️  未找到有效的文档分块")
        else:
            print("📁 未找到文档，尝试加载已有索引...")
            index_path = "data/vectors/main_index"
            if Path(f"{index_path}.faiss").exists():
                retriever.load_index(index_path)
                print("✅ 已加载现有索引")
            else:
                print("⚠️  未找到现有索引，系统将以演示模式运行")
        
        # 初始化对话服务
        print("💬 初始化对话服务...")
        chat_service = ChatService(config_manager)
        
        logger.info("系统初始化完成")
        print("✅ 系统准备就绪！")
        
        # 3. 启动交互式问答循环
        print("\n" + "=" * 50)
        print("🎯 RAG学习系统 - 交互式问答")
        print("=" * 50)
        print("💡 输入问题开始对话，输入 'quit' 或 'exit' 退出")
        print("💡 输入 'help' 查看帮助信息")
        print("-" * 50)
        
        while True:
            try:
                # 获取用户输入
                question = input("\n🤔 请输入您的问题: ").strip()
                
                if not question:
                    continue
                
                # 处理特殊命令
                if question.lower() in ['quit', 'exit', '退出']:
                    print("👋 感谢使用RAG学习系统，再见！")
                    break
                elif question.lower() in ['help', '帮助']:
                    print("\n📖 帮助信息:")
                    print("  • 直接输入问题进行智能问答")
                    print("  • 系统会基于知识库内容生成回答")
                    print("  • 输入 'quit' 或 'exit' 退出系统")
                    print("  • 输入 'stats' 查看系统统计信息")
                    continue
                elif question.lower() in ['stats', '统计']:
                    stats = retriever.get_stats()
                    print(f"\n📊 系统统计:")
                    print(f"  • 索引状态: {'已构建' if stats['index_built'] else '未构建'}")
                    print(f"  • 文档分块数: {stats['chunk_count']}")
                    print(f"  • 向量维度: {stats.get('vector_dimension', 'N/A')}")
                    continue
                
                # 生成回答（使用流式输出）
                print("🤖 正在思考...")
                start_time = time.time()
                
                # 使用流式生成答案
                sources = []
                confidence = 0.0
                full_answer = ""
                response_time = 0.0
                error_msg = None
                
                print(f"\n💡 回答:")
                
                try:
                    for chunk_data in chat_service.generate_answer_stream(question, retriever):
                        if chunk_data['type'] == 'start':
                            sources = chunk_data['sources']
                            confidence = chunk_data['confidence']
                        elif chunk_data['type'] == 'chunk':
                            content = chunk_data['content']
                            print(content, end='', flush=True)  # 实时显示文本块
                            full_answer += content
                        elif chunk_data['type'] == 'end':
                            response_time = chunk_data['response_time']
                        elif chunk_data['type'] == 'error':
                            error_msg = chunk_data['error']
                            if 'content' in chunk_data:
                                print(chunk_data['content'])
                            break
                    
                    print()  # 换行
                    
                    # 显示元信息
                    if sources:
                        print(f"\n📚 参考来源: {', '.join(sources)}")
                    
                    print(f"🎯 置信度: {confidence:.3f}")
                    print(f"⏱️  响应时间: {response_time:.2f}秒")
                    
                    if error_msg:
                        print(f"⚠️  警告: {error_msg}")
                        
                except Exception as e:
                    print(f"\n❌ 流式输出错误: {e}")
                    print("💡 请检查网络连接或稍后重试")
                
            except KeyboardInterrupt:
                print("\n\n👋 检测到中断信号，正在退出...")
                break
            except Exception as e:
                logger.error(f"问答过程中发生错误: {e}")
                print(f"❌ 发生错误: {e}")
                print("💡 请重试或输入 'quit' 退出")
        
    except FileNotFoundError as e:
        logger.error(f"配置文件错误: {e}")
        print(f"❌ 配置文件错误: {e}")
        print("💡 请检查config.yaml文件是否存在")
        return 1
    except ValueError as e:
        logger.error(f"配置验证失败: {e}")
        print(f"❌ 配置验证失败: {e}")
        print("💡 请检查配置文件格式和必需参数")
        return 1
    except Exception as e:
        logger.error(f"系统启动失败: {e}")
        print(f"❌ 启动失败: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())