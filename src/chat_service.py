"""
对话服务模块

提供基于RAG的智能问答服务，集成DeepSeek API
只支持流式输出，简化接口设计
"""

import logging
import time
from typing import List, Optional, Dict, Any, Generator
from openai import OpenAI

from .config_manager import ConfigManager
from .retriever import RAGRetriever

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ChatService:
    """对话服务类"""
    
    def __init__(self, config_manager: ConfigManager):
        """
        初始化对话服务
        
        Args:
            config_manager: 配置管理器实例
        """
        self.config_manager = config_manager
        self.config = config_manager.get_config()
        self.client = None
        self._initialize_client()
        
        # 从配置中获取模型参数
        self.model_name = self.config.llm_model
        self.max_tokens = self.config.llm_max_tokens
        self.temperature = self.config.llm_temperature
        self.max_retries = 3  # 默认重试次数
        self.retry_delay = 1  # 默认重试延迟
        
        logger.info(f"ChatService初始化完成，使用模型: {self.model_name}")
    
    def _initialize_client(self):
        """初始化DeepSeek API客户端"""
        try:
            api_key = self.config.llm_api_key
            base_url = self.config.llm_base_url
            
            if not api_key:
                raise ValueError("DeepSeek API密钥未配置")
            
            self.client = OpenAI(
                api_key=api_key,
                base_url=base_url
            )
            
            logger.info("DeepSeek API客户端初始化成功")
            
        except Exception as e:
            logger.error(f"DeepSeek API客户端初始化失败: {e}")
            raise
    

    
    def generate_answer_stream(self, question: str, retriever: RAGRetriever) -> Generator[Dict[str, Any], None, None]:
        """
        生成问题的答案（流式）
        
        Args:
            question: 用户问题
            retriever: RAG检索器实例
            
        Yields:
            Dict[str, Any]: 包含流式数据的字典
                - type: 'start' | 'chunk' | 'end' | 'error'
                - content: 文本内容（仅在type='chunk'时）
                - sources: 参考来源（仅在type='start'时）
                - confidence: 置信度（仅在type='start'时）
                - response_time: 响应时间（仅在type='end'时）
                - error: 错误信息（仅在type='error'时）
        """
        start_time = time.time()
        
        try:
            logger.info(f"开始流式处理问题: {question[:50]}...")
            
            # 1. 使用RAG检索相关文档
            top_k = self.config.retrieval_top_k
            similarity_threshold = self.config.retrieval_similarity_threshold
            
            logger.info(f"检索参数: top_k={top_k}, similarity_threshold={similarity_threshold}")
            search_results = retriever.search(question, top_k=top_k, similarity_threshold=similarity_threshold)
            
            if not search_results:
                logger.warning("未找到相关文档")
                yield {
                    'type': 'error',
                    'error': '未找到相关文档',
                    'content': '抱歉，我在知识库中没有找到相关信息来回答您的问题。'
                }
                return
            
            # 2. 提取上下文和来源
            context_texts = []
            sources = []
            
            for result in search_results:
                context_texts.append(result.content)
                source = result.metadata.get('source', '未知来源')
                if source not in sources:
                    sources.append(source)
            
            context = "\n\n".join(context_texts)
            confidence = sum(result.score for result in search_results) / len(search_results)
            
            # 3. 发送开始信号
            yield {
                'type': 'start',
                'sources': sources,
                'confidence': confidence
            }
            
            # 4. 构建提示词
            prompt = self.build_prompt(question, context)
            
            # 5. 流式调用API
            for chunk in self.call_api_stream(prompt):
                yield {
                    'type': 'chunk',
                    'content': chunk
                }
            
            # 6. 发送结束信号
            response_time = time.time() - start_time
            logger.info(f"流式问题处理完成，耗时: {response_time:.2f}秒")
            
            yield {
                'type': 'end',
                'response_time': response_time
            }
            
        except Exception as e:
            error_msg = f"生成答案时发生错误: {str(e)}"
            logger.error(error_msg)
            
            yield {
                'type': 'error',
                'error': error_msg,
                'content': '抱歉，处理您的问题时遇到了技术问题，请稍后重试。'
            }
    
    def build_prompt(self, question: str, context: str) -> str:
        """
        构建RAG提示词
        
        Args:
            question: 用户问题
            context: 检索到的相关文档内容
            
        Returns:
            str: 构建好的提示词
        """
        prompt_template = """你是一个专业的AI助手，请基于以下提供的知识库内容来回答用户的问题。

知识库内容：
{context}

用户问题：{question}

请遵循以下要求：
1. 仅基于提供的知识库内容回答问题
2. 如果知识库中没有相关信息，请明确说明
3. 回答要准确、详细且有条理
4. 使用中文回答
5. 如果可能，请提供具体的例子或解释

回答："""
        
        return prompt_template.format(context=context, question=question)
    

    def call_api_stream(self, prompt: str) -> Generator[str, None, None]:
        """
        流式调用DeepSeek API
        
        Args:
            prompt: 构建好的提示词
            
        Yields:
            str: API返回的文本块
        """
        for attempt in range(self.max_retries):
            try:
                logger.debug(f"流式调用DeepSeek API，尝试次数: {attempt + 1}")
                
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                    stream=True  # 启用流式输出
                )
                
                logger.debug("DeepSeek API流式调用成功")
                
                # 处理流式响应
                for chunk in response:
                    if chunk.choices[0].delta.content is not None:
                        content = chunk.choices[0].delta.content
                        yield content
                
                return  # 成功完成，退出重试循环
                
            except Exception as e:
                logger.warning(f"流式API调用失败 (尝试 {attempt + 1}/{self.max_retries}): {e}")
                
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (attempt + 1))  # 指数退避
                else:
                    raise Exception(f"流式API调用失败，已重试{self.max_retries}次: {str(e)}")
    
    def health_check(self) -> Dict[str, Any]:
        """
        健康检查
        
        Returns:
            Dict[str, Any]: 健康状态信息
        """
        try:
            # 测试API连接
            test_response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": "Hello"}],
                max_tokens=10
            )
            
            return {
                "status": "healthy",
                "model": self.model_name,
                "api_accessible": True,
                "error": None
            }
            
        except Exception as e:
            return {
                "status": "unhealthy",
                "model": self.model_name,
                "api_accessible": False,
                "error": str(e)
            }