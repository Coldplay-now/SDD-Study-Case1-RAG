#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RAG学习系统 - 综合功能测试脚本
测试系统的完整问答功能，验证检索和生成质量
"""

import time
import subprocess
import threading
import queue
import sys
from typing import List, Dict, Any

class RAGSystemTester:
    """RAG系统测试器"""
    
    def __init__(self):
        self.test_questions = [
            "什么是机器学习？请详细解释其基本概念。",
            "深度学习和传统机器学习有什么区别？",
            "什么是RAG技术？它是如何工作的？",
            "人工智能在哪些领域有重要应用？",
            "神经网络的基本原理是什么？",
            "监督学习和无监督学习的区别是什么？"
        ]
        
    def test_interactive_qa(self):
        """测试交互式问答功能"""
        print("🧪 开始RAG系统综合功能测试")
        print("=" * 60)
        
        print("\n📋 测试问题列表：")
        for i, question in enumerate(self.test_questions, 1):
            print(f"  {i}. {question}")
        
        print("\n" + "=" * 60)
        print("🎯 测试指导：")
        print("1. 请在终端4中手动输入以下测试问题")
        print("2. 观察系统的检索和回答过程")
        print("3. 验证回答的相关性和准确性")
        print("4. 检查系统性能指标")
        print("=" * 60)
        
        print("\n🚀 开始逐个测试...")
        
        for i, question in enumerate(self.test_questions, 1):
            print(f"\n📝 测试问题 {i}/{len(self.test_questions)}:")
            print(f"   {question}")
            print("\n⏳ 请在终端4输入上述问题，然后按Enter继续下一个测试...")
            
            # 等待用户确认
            input("   按Enter键继续下一个测试...")
        
        print("\n✅ 所有测试问题已列出完成！")
        print("\n📊 请验证以下功能点：")
        print("   ✓ 文档检索功能是否正常")
        print("   ✓ 回答生成是否基于检索内容")
        print("   ✓ 回答质量是否满足要求")
        print("   ✓ 系统响应时间是否合理")
        print("   ✓ 错误处理是否完善")
        
    def check_system_logs(self):
        """检查系统日志"""
        print("\n📋 检查系统运行日志...")
        try:
            result = subprocess.run(
                ["tail", "-20", "rag_system.log"],
                capture_output=True,
                text=True,
                cwd="/Users/xt/Desktop/code/trae/921_Rag1/rag_learning_system"
            )
            
            if result.returncode == 0:
                print("📄 最新系统日志：")
                print("-" * 40)
                print(result.stdout)
                print("-" * 40)
            else:
                print("❌ 无法读取系统日志")
                
        except Exception as e:
            print(f"❌ 日志检查失败: {e}")
    
    def performance_summary(self):
        """性能总结"""
        print("\n📈 性能验证要点：")
        print("1. 系统启动时间: 应 < 30秒")
        print("2. 单次问答响应: 应 < 10秒")
        print("3. 检索响应时间: 应 < 500ms")
        print("4. 内存使用情况: 观察是否稳定")
        print("5. API调用成功率: 应接近100%")

def main():
    """主测试函数"""
    tester = RAGSystemTester()
    
    print("🎯 RAG学习系统 - 综合功能测试")
    print("测试目标: 验证系统完整的问答功能")
    print("测试方式: 手动交互测试 + 日志分析")
    
    # 执行测试
    tester.test_interactive_qa()
    tester.check_system_logs()
    tester.performance_summary()
    
    print("\n🎉 测试完成！")
    print("请根据测试结果评估系统功能是否满足要求。")

if __name__ == "__main__":
    main()