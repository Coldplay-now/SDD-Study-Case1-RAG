"""
简化的配置管理器
版本: v2.0 - 优化版

主要改进:
- 简化数据类结构 (6个 -> 2个)
- 移除复杂的配置合并逻辑
- 保持API兼容性
- 单一配置文件支持
"""

import os
import yaml
import logging
from dataclasses import dataclass, field
from typing import Dict, Any, Optional
from pathlib import Path
from dotenv import load_dotenv

logger = logging.getLogger(__name__)


@dataclass
class Config:
    """统一的配置数据类"""
    
    # 嵌入模型配置
    embedding_model_name: str = "BAAI/bge-small-zh-v1.5"
    embedding_device: str = "cpu"
    embedding_max_length: int = 512
    embedding_batch_size: int = 32
    
    # 检索配置
    retrieval_chunk_size: int = 500
    retrieval_chunk_overlap: int = 50
    retrieval_top_k: int = 5
    retrieval_similarity_threshold: float = 0.3
    
    # LLM配置
    llm_model: str = "deepseek-chat"
    llm_max_tokens: int = 1000
    llm_temperature: float = 0.7
    llm_stream: bool = True
    llm_api_key: str = ""
    llm_base_url: str = "https://api.deepseek.com"
    
    # 路径配置
    documents_path: str = "./data/documents"
    vectors_path: str = "./data/vectors"
    cache_path: str = "./data/cache"
    
    # 系统配置
    log_level: str = "INFO"
    cache_enabled: bool = True
    max_documents: int = 100
    
    def __post_init__(self):
        """配置后处理和验证"""
        # 确保路径存在
        for path_attr in ['documents_path', 'vectors_path', 'cache_path']:
            path = getattr(self, path_attr)
            Path(path).mkdir(parents=True, exist_ok=True)
    
    # 为了保持向后兼容性，提供属性访问方法
    @property
    def embedding(self) -> Dict[str, Any]:
        """返回嵌入配置字典（向后兼容）"""
        return {
            'model_name': self.embedding_model_name,
            'device': self.embedding_device,
            'max_length': self.embedding_max_length,
            'batch_size': self.embedding_batch_size
        }
    
    @property
    def retrieval(self) -> Dict[str, Any]:
        """返回检索配置字典（向后兼容）"""
        return {
            'chunk_size': self.retrieval_chunk_size,
            'chunk_overlap': self.retrieval_chunk_overlap,
            'top_k': self.retrieval_top_k,
            'similarity_threshold': self.retrieval_similarity_threshold
        }
    
    @property
    def llm(self) -> Dict[str, Any]:
        """返回LLM配置字典（向后兼容）"""
        return {
            'model': self.llm_model,
            'max_tokens': self.llm_max_tokens,
            'temperature': self.llm_temperature,
            'stream': self.llm_stream,
            'api_key': self.llm_api_key,
            'base_url': self.llm_base_url
        }
    
    @property
    def paths(self) -> Dict[str, str]:
        """返回路径配置字典（向后兼容）"""
        return {
            'documents': self.documents_path,
            'vectors': self.vectors_path,
            'cache': self.cache_path
        }
    
    @property
    def system(self) -> Dict[str, Any]:
        """返回系统配置字典（向后兼容）"""
        return {
            'log_level': self.log_level,
            'cache_enabled': self.cache_enabled,
            'max_documents': self.max_documents
        }


class ConfigManager:
    """简化的配置管理器"""
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        初始化配置管理器
        
        Args:
            config_path: 配置文件路径
        """
        self.config_path = config_path
        self._config: Optional[Config] = None
        
    def load_config(self) -> Config:
        """
        加载配置文件
        
        Returns:
            Config: 配置对象
        """
        if self._config is not None:
            return self._config
            
        try:
            # 加载环境变量
            self._load_env_variables()
            
            # 加载YAML配置文件
            config_data = self._load_yaml_config()
            
            # 创建配置对象
            self._config = self._create_config_from_dict(config_data)
            
            logger.info(f"配置加载成功: {self.config_path}")
            return self._config
            
        except Exception as e:
            logger.error(f"配置加载失败: {e}")
            # 返回默认配置
            self._config = Config()
            logger.info("使用默认配置")
            return self._config
    
    def _load_env_variables(self):
        """加载环境变量"""
        env_path = Path(".env")
        if env_path.exists():
            load_dotenv(env_path)
            logger.info(f"环境变量文件加载成功: {env_path}")
        else:
            logger.warning(f"环境变量文件不存在: {env_path}")
    
    def _load_yaml_config(self) -> Dict[str, Any]:
        """加载YAML配置文件"""
        if not os.path.exists(self.config_path):
            logger.warning(f"配置文件不存在: {self.config_path}")
            return {}
            
        with open(self.config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f) or {}
    
    def _create_config_from_dict(self, config_data: Dict[str, Any]) -> Config:
        """从字典创建配置对象"""
        # 扁平化嵌套配置
        flat_config = {}
        
        # 处理嵌入配置
        if 'embedding' in config_data:
            embedding = config_data['embedding']
            flat_config.update({
                'embedding_model_name': embedding.get('model_name', 'BAAI/bge-small-zh-v1.5'),
                'embedding_device': embedding.get('device', 'cpu'),
                'embedding_max_length': embedding.get('max_length', 512),
                'embedding_batch_size': embedding.get('batch_size', 32)
            })
        
        # 处理检索配置
        if 'retrieval' in config_data:
            retrieval = config_data['retrieval']
            flat_config.update({
                'retrieval_chunk_size': retrieval.get('chunk_size', 500),
                'retrieval_chunk_overlap': retrieval.get('chunk_overlap', 50),
                'retrieval_top_k': retrieval.get('top_k', 5),
                'retrieval_similarity_threshold': retrieval.get('similarity_threshold', 0.3)
            })
        
        # 处理LLM配置
        if 'llm' in config_data:
            llm = config_data['llm']
            flat_config.update({
                'llm_model': llm.get('model', 'deepseek-chat'),
                'llm_max_tokens': llm.get('max_tokens', 1000),
                'llm_temperature': llm.get('temperature', 0.7),
                'llm_stream': llm.get('stream', True),
                'llm_api_key': llm.get('api_key', ''),
                'llm_base_url': llm.get('base_url', 'https://api.deepseek.com')
            })
        
        # 环境变量覆盖（优先级更高）
        flat_config.update({
            'llm_api_key': os.getenv('DEEPSEEK_API_KEY', flat_config.get('llm_api_key', '')),
            'llm_base_url': os.getenv('DEEPSEEK_BASE_URL', flat_config.get('llm_base_url', 'https://api.deepseek.com')),
            'log_level': os.getenv('LOG_LEVEL', flat_config.get('log_level', 'INFO')),
            'cache_enabled': os.getenv('CACHE_ENABLED', str(flat_config.get('cache_enabled', True))).lower() in ('true', '1', 'yes', 'on')
        })
        
        # 处理路径配置
        if 'paths' in config_data:
            paths = config_data['paths']
            flat_config.update({
                'documents_path': paths.get('documents', './data/documents'),
                'vectors_path': paths.get('vectors', './data/vectors'),
                'cache_path': paths.get('cache', './data/cache')
            })
        
        # 处理系统配置
        if 'system' in config_data:
            system = config_data['system']
            flat_config.update({
                'log_level': system.get('log_level', 'INFO'),
                'cache_enabled': system.get('cache_enabled', True),
                'max_documents': system.get('max_documents', 100)
            })
        
        return Config(**flat_config)
    
    def get_config(self) -> Config:
        """获取配置对象"""
        if self._config is None:
            return self.load_config()
        return self._config
    
    def reload_config(self) -> Config:
        """重新加载配置"""
        self._config = None
        return self.load_config()


# 为了保持向后兼容性，提供旧的类名别名
EmbeddingConfig = Config
RetrievalConfig = Config
LLMConfig = Config
PathsConfig = Config
SystemConfig = Config
AppConfig = Config


def get_config_manager(config_path: str = "config.yaml") -> ConfigManager:
    """获取配置管理器实例"""
    return ConfigManager(config_path)


def get_config(config_path: str = "config.yaml") -> Config:
    """快速获取配置对象"""
    manager = ConfigManager(config_path)
    return manager.load_config()