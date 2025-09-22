#!/bin/bash

# RAG学习系统启动脚本
# 作者: RAG系统开发团队
# 版本: 1.0

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
WHITE='\033[1;37m'
NC='\033[0m' # No Color

# 清屏
clear

# 显示ASCII艺术字 "RAG NOW"
echo -e "${CYAN}"
cat << "EOF"
██████╗  █████╗  ██████╗     ███╗   ██╗ ██████╗ ██╗    ██╗
██╔══██╗██╔══██╗██╔════╝     ████╗  ██║██╔═══██╗██║    ██║
██████╔╝███████║██║  ███╗    ██╔██╗ ██║██║   ██║██║ █╗ ██║
██╔══██╗██╔══██║██║   ██║    ██║╚██╗██║██║   ██║██║███╗██║
██║  ██║██║  ██║╚██████╔╝    ██║ ╚████║╚██████╔╝╚███╔███╔╝
╚═╝  ╚═╝╚═╝  ╚═╝ ╚═════╝     ╚═╝  ╚═══╝ ╚═════╝  ╚══╝╚══╝ 
EOF
echo -e "${NC}"

# 显示欢迎信息
echo -e "${WHITE}================================================================${NC}"
echo -e "${GREEN}🎉 欢迎使用 RAG 智能学习系统！${NC}"
echo -e "${WHITE}================================================================${NC}"
echo ""

# 显示系统信息
echo -e "${YELLOW}📋 系统特性:${NC}"
echo -e "  ${BLUE}🤖 智能问答${NC} - 基于深度学习的自然语言理解"
echo -e "  ${BLUE}📚 知识检索${NC} - 高效的向量化文档搜索"
echo -e "  ${BLUE}💬 流式输出${NC} - 实时响应，流畅体验"
echo -e "  ${BLUE}🎯 精准匹配${NC} - 智能相似度计算"
echo ""

# 显示使用指南
echo -e "${YELLOW}💡 使用指南:${NC}"
echo -e "  ${GREEN}• 输入任何问题开始智能对话${NC}"
echo -e "  ${GREEN}• 输入 'help' 查看详细帮助${NC}"
echo -e "  ${GREEN}• 输入 'stats' 查看系统统计${NC}"
echo -e "  ${GREEN}• 输入 'quit' 或 'exit' 退出系统${NC}"
echo ""

# 显示示例问题
echo -e "${YELLOW}🔍 示例问题:${NC}"
echo -e "  ${PURPLE}• 什么是机器学习？${NC}"
echo -e "  ${PURPLE}• 深度学习的基本原理是什么？${NC}"
echo -e "  ${PURPLE}• 如何选择合适的算法？${NC}"
echo -e "  ${PURPLE}• 数据预处理的重要性${NC}"
echo ""

echo -e "${WHITE}================================================================${NC}"

# 检查Python环境
echo -e "${YELLOW}🔧 正在检查系统环境...${NC}"

if ! command -v python3 &> /dev/null; then
    echo -e "${RED}❌ 错误: 未找到 Python3，请先安装 Python3${NC}"
    exit 1
fi

# 检查虚拟环境
if [ ! -d "venv" ]; then
    echo -e "${YELLOW}⚠️  未找到虚拟环境，正在创建...${NC}"
    python3 -m venv venv
    echo -e "${GREEN}✅ 虚拟环境创建成功${NC}"
fi

# 激活虚拟环境
echo -e "${YELLOW}🔄 正在激活虚拟环境...${NC}"
source venv/bin/activate

# 检查依赖
if [ ! -f "requirements.txt" ]; then
    echo -e "${RED}❌ 错误: 未找到 requirements.txt 文件${NC}"
    exit 1
fi

# 安装依赖（如果需要）
echo -e "${YELLOW}📦 正在检查依赖包...${NC}"
pip install -q -r requirements.txt

# 检查配置文件
if [ ! -f "config.yaml" ]; then
    echo -e "${RED}❌ 错误: 未找到配置文件 config.yaml${NC}"
    exit 1
fi

if [ ! -f ".env" ]; then
    echo -e "${RED}❌ 错误: 未找到环境配置文件 .env${NC}"
    echo -e "${YELLOW}💡 请参考 .env.example 创建 .env 文件${NC}"
    exit 1
fi

echo -e "${GREEN}✅ 环境检查完成${NC}"
echo ""

# 显示启动信息
echo -e "${WHITE}================================================================${NC}"
echo -e "${GREEN}🚀 正在启动 RAG 智能学习系统...${NC}"
echo -e "${WHITE}================================================================${NC}"
echo ""

# 启动主程序
python main.py

# 程序结束后的清理
echo ""
echo -e "${YELLOW}👋 感谢使用 RAG 智能学习系统！${NC}"
echo -e "${BLUE}💡 如有问题，请查看日志文件或联系技术支持${NC}"
echo ""