#!/usr/bin/env python3
"""
文档处理模块测试脚本
"""

import sys
import os
from pathlib import Path

# 添加src目录到Python路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from document_processor import DocumentProcessor

def test_document_processor():
    """测试文档处理器功能"""
    print("=== 文档处理模块测试 ===\n")
    
    # 初始化处理器
    processor = DocumentProcessor(
        chunk_size=500,
        chunk_overlap=50
    )
    
    # 测试文档路径
    test_doc_path = "data/documents/test_document.md"
    
    if not os.path.exists(test_doc_path):
        print(f"❌ 测试文档不存在: {test_doc_path}")
        return
    
    try:
        print(f"📄 加载文档: {test_doc_path}")
        
        # 加载文档
        content = processor.load_document(test_doc_path)
        print(f"✅ 文档加载成功，长度: {len(content)} 字符\n")
        
        # 处理文档
        print("🔄 开始处理文档...")
        chunks = processor.process_document(test_doc_path)
        
        print(f"✅ 文档处理完成，生成 {len(chunks)} 个分块\n")
        
        # 显示分块信息
        print("📊 分块详情:")
        print("-" * 80)
        
        for i, chunk in enumerate(chunks[:5]):  # 只显示前5个分块
            print(f"分块 {chunk.chunk_index + 1}:")
            print(f"  - 长度: {len(chunk.content)} 字符")
            print(f"  - 来源: {chunk.metadata.get('source', test_doc_path)}")
            print(f"  - 标题: {chunk.metadata.get('title', 'N/A')}")
            print(f"  - 内容预览: {chunk.content[:100]}...")
            print()
        
        if len(chunks) > 5:
            print(f"... 还有 {len(chunks) - 5} 个分块")
        
        print("-" * 80)
        
        # 统计信息
        total_chars = sum(len(chunk.content) for chunk in chunks)
        avg_chunk_size = total_chars / len(chunks) if chunks else 0
        
        print(f"\n📈 统计信息:")
        print(f"  - 总分块数: {len(chunks)}")
        print(f"  - 总字符数: {total_chars}")
        print(f"  - 平均分块大小: {avg_chunk_size:.1f} 字符")
        print(f"  - 原文档大小: {len(content)} 字符")
        print(f"  - 处理效率: {(total_chars/len(content)*100):.1f}%")
        
    except Exception as e:
        print(f"❌ 测试失败: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_document_processor()