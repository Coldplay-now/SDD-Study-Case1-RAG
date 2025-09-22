#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
相似度阈值过滤功能测试脚本
"""

import sys
import logging
from pathlib import Path

# 添加src目录到Python路径
sys.path.append(str(Path(__file__).parent / "src"))

from config_manager import ConfigManager
from src.embedding_model import EmbeddingModel
from src.retriever import RAGRetriever

def test_similarity_threshold():
    """测试相似度阈值过滤功能"""
    print("=" * 60)
    print("🧪 相似度阈值过滤功能测试")
    print("=" * 60)
    
    # 设置日志
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    try:
        # 1. 加载配置
        config_manager = ConfigManager()
        config = config_manager.load_config()
        
        print(f"📊 当前配置:")
        print(f"  • 相似度阈值: {config.retrieval.similarity_threshold}")
        print(f"  • 检索数量: {config.retrieval.top_k}")
        
        # 2. 初始化服务
        embedding_service = EmbeddingService(
            model_name=config.embedding.model_name,
            device=config.embedding.device
        )
        embedding_service.load_model()
        
        retriever = RAGRetriever(embedding_service=embedding_service)
        
        # 3. 加载现有索引
        index_path = "data/vectors/main_index"
        if Path(f"{index_path}.faiss").exists():
            retriever.load_index(index_path)
            print("✅ 已加载现有索引")
        else:
            print("❌ 未找到索引文件")
            return
        
        # 4. 测试不同类型的查询
        test_queries = [
            {
                "query": "什么是机器学习？",
                "description": "高相关性查询（应该有结果）"
            },
            {
                "query": "深度学习的特点",
                "description": "中等相关性查询"
            },
            {
                "query": "今天天气怎么样？",
                "description": "低相关性查询（可能被过滤）"
            },
            {
                "query": "如何做红烧肉？",
                "description": "无关查询（应该被过滤）"
            }
        ]
        
        print("\n🔍 开始测试不同查询的过滤效果:")
        print("-" * 60)
        
        for i, test_case in enumerate(test_queries, 1):
            query = test_case["query"]
            description = test_case["description"]
            
            print(f"\n测试 {i}: {description}")
            print(f"查询: {query}")
            
            # 使用默认阈值检索
            results = retriever.search(
                query, 
                top_k=config.retrieval.top_k,
                similarity_threshold=config.retrieval.similarity_threshold
            )
            
            print(f"结果数量: {len(results)}")
            
            if results:
                print("检索结果:")
                for j, result in enumerate(results, 1):
                    print(f"  {j}. 相似度: {result.score:.3f}")
                    print(f"     内容: {result.content[:100]}...")
            else:
                print("  ❌ 无结果（被相似度阈值过滤）")
            
            # 测试不同阈值的效果
            print(f"\n🔬 阈值对比测试:")
            for threshold in [0.1, 0.3, 0.5, 0.7]:
                test_results = retriever.search(query, top_k=5, similarity_threshold=threshold)
                print(f"  阈值 {threshold}: {len(test_results)} 个结果")
        
        print("\n" + "=" * 60)
        print("✅ 相似度阈值过滤功能测试完成")
        print("=" * 60)
        
        # 5. 性能建议
        print("\n💡 配置建议:")
        print("  • 阈值 0.1-0.3: 宽松过滤，更多结果")
        print("  • 阈值 0.3-0.5: 平衡过滤，推荐设置")
        print("  • 阈值 0.5-0.7: 严格过滤，高质量结果")
        print("  • 阈值 > 0.7: 极严格过滤，可能过度限制")
        
    except Exception as e:
        logger.error(f"测试失败: {e}")
        print(f"❌ 测试失败: {e}")

if __name__ == "__main__":
    test_similarity_threshold()