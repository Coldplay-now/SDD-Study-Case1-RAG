#!/usr/bin/env python3
"""
RAG系统自动化测试脚本
使用subprocess和管道进行自动化交互测试
"""

import subprocess
import time
import threading
import queue
import sys
import os

def run_rag_test():
    """运行RAG系统并进行自动化测试"""
    print("🚀 启动RAG系统自动化测试...")
    
    # 测试问题
    test_questions = [
        "什么是机器学习？",
        "quit"  # 退出命令
    ]
    
    try:
        # 启动RAG系统进程
        print("📱 启动RAG系统进程...")
        process = subprocess.Popen(
            ["python", "main.py"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            universal_newlines=True,
            cwd="/Users/xt/Desktop/code/trae/921_Rag1/rag_learning_system"
        )
        
        print("⏳ 等待系统初始化...")
        time.sleep(10)  # 等待系统完全启动
        
        print("🔍 开始发送测试问题...")
        
        for i, question in enumerate(test_questions, 1):
            if question == "quit":
                print(f"\n📤 发送退出命令: {question}")
            else:
                print(f"\n📤 发送测试问题 {i}: {question}")
            
            # 发送问题
            process.stdin.write(question + "\n")
            process.stdin.flush()
            
            if question != "quit":
                print("⏳ 等待系统响应...")
                time.sleep(5)  # 等待响应
        
        # 等待进程结束
        print("\n⏳ 等待进程结束...")
        stdout, stderr = process.communicate(timeout=30)
        
        print("\n📊 测试结果:")
        print("=" * 50)
        print("标准输出:")
        print(stdout)
        
        if stderr:
            print("\n错误输出:")
            print(stderr)
        
        print(f"\n进程退出码: {process.returncode}")
        
    except subprocess.TimeoutExpired:
        print("⚠️ 进程超时，强制终止...")
        process.kill()
        stdout, stderr = process.communicate()
        print("超时前的输出:")
        print(stdout)
        
    except Exception as e:
        print(f"❌ 测试过程中出错: {e}")
        
    print("\n✅ 自动化测试完成！")

if __name__ == "__main__":
    # 确保在正确的目录中运行
    os.chdir("/Users/xt/Desktop/code/trae/921_Rag1/rag_learning_system")
    
    # 激活虚拟环境并运行测试
    print("🔧 激活虚拟环境...")
    os.system("source venv/bin/activate")
    
    run_rag_test()