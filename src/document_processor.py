#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
文档处理模块 - Markdown文档专用
负责Markdown文档的加载、分块和预处理
"""

import os
import re
import logging
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime

@dataclass
class DocumentChunk:
    """文档分块数据结构"""
    id: str
    content: str
    metadata: Dict[str, Any]
    source_file: str
    chunk_index: int
    start_pos: int
    end_pos: int

class MarkdownDocumentProcessor:
    """Markdown文档处理器"""
    
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        """
        初始化Markdown文档处理器
        
        Args:
            chunk_size: 分块大小（字符数）
            chunk_overlap: 分块重叠大小（字符数）
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.logger = logging.getLogger(__name__)
        
        # Markdown结构正则表达式
        self.header_pattern = re.compile(r'^(#{1,6})\s+(.+)$', re.MULTILINE)
        self.code_block_pattern = re.compile(r'```[\s\S]*?```', re.MULTILINE)
        self.list_pattern = re.compile(r'^[\s]*[-*+]\s+', re.MULTILINE)
        
    def load_document(self, file_path: str) -> str:
        """
        加载Markdown文档内容
        
        Args:
            file_path: 文档路径
            
        Returns:
            文档内容
            
        Raises:
            FileNotFoundError: 文件不存在
            ValueError: 不是Markdown文件
            UnicodeDecodeError: 文件编码错误
        """
        file_path = Path(file_path)
        
        # 验证文件存在
        if not file_path.exists():
            raise FileNotFoundError(f"文档文件不存在: {file_path}")
        
        # 验证文件格式
        if file_path.suffix.lower() != '.md':
            raise ValueError(f"不支持的文件格式: {file_path.suffix}，仅支持.md文件")
        
        try:
            # 尝试多种编码读取文件
            encodings = ['utf-8', 'utf-8-sig', 'gbk', 'gb2312']
            content = None
            
            for encoding in encodings:
                try:
                    with open(file_path, 'r', encoding=encoding) as f:
                        content = f.read()
                    self.logger.info(f"成功使用{encoding}编码读取文件: {file_path}")
                    break
                except UnicodeDecodeError:
                    continue
            
            if content is None:
                raise UnicodeDecodeError("无法使用常见编码读取文件")
            
            return content
            
        except Exception as e:
            self.logger.error(f"加载文档失败: {file_path}, 错误: {e}")
            raise
    
    def process_document(self, file_path: str) -> List[DocumentChunk]:
        """
        处理Markdown文档，返回分块结果
        
        Args:
            file_path: 文档路径
            
        Returns:
            文档分块列表
        """
        try:
            # 加载文档内容
            content = self.load_document(file_path)
            
            # 预处理文档
            processed_content = self._preprocess_markdown(content)
            
            # 提取文档元数据
            metadata = self._extract_metadata(file_path, content)
            
            # 分块处理
            chunks = self._chunk_markdown_text(processed_content, metadata, file_path)
            
            self.logger.info(f"文档处理完成: {file_path}, 生成{len(chunks)}个分块")
            return chunks
            
        except Exception as e:
            self.logger.error(f"处理文档失败: {file_path}, 错误: {e}")
            raise
    
    def _preprocess_markdown(self, content: str) -> str:
        """
        预处理Markdown内容
        
        Args:
            content: 原始内容
            
        Returns:
            预处理后的内容
        """
        # 移除多余的空行（保留段落结构）
        content = re.sub(r'\n\s*\n\s*\n', '\n\n', content)
        
        # 清理行尾空格
        content = re.sub(r'[ \t]+$', '', content, flags=re.MULTILINE)
        
        # 确保文档以换行符结尾
        if not content.endswith('\n'):
            content += '\n'
        
        return content.strip()
    
    def _extract_metadata(self, file_path: str, content: str) -> Dict[str, Any]:
        """
        提取文档元数据
        
        Args:
            file_path: 文件路径
            content: 文档内容
            
        Returns:
            元数据字典
        """
        file_path = Path(file_path)
        stat = file_path.stat()
        
        # 提取标题（第一个一级标题）
        title_match = re.search(r'^#\s+(.+)$', content, re.MULTILINE)
        title = title_match.group(1) if title_match else file_path.stem
        
        # 统计文档信息
        headers = self.header_pattern.findall(content)
        code_blocks = self.code_block_pattern.findall(content)
        
        metadata = {
            'file_name': file_path.name,
            'file_path': str(file_path.absolute()),
            'file_size': stat.st_size,
            'created_time': datetime.fromtimestamp(stat.st_ctime),
            'modified_time': datetime.fromtimestamp(stat.st_mtime),
            'title': title,
            'content_length': len(content),
            'header_count': len(headers),
            'code_block_count': len(code_blocks),
            'document_type': 'markdown'
        }
        
        return metadata
    
    def _chunk_markdown_text(self, text: str, metadata: Dict[str, Any], file_path: str) -> List[DocumentChunk]:
        """
        智能分块Markdown文本
        
        Args:
            text: 文本内容
            metadata: 文档元数据
            file_path: 文件路径
            
        Returns:
            分块列表
        """
        chunks = []
        
        # 按段落分割（保持Markdown结构）
        paragraphs = self._split_by_structure(text)
        
        current_chunk = ""
        current_start = 0
        chunk_index = 0
        
        for paragraph in paragraphs:
            # 如果当前段落本身就超过chunk_size，需要强制分割
            if len(paragraph) > self.chunk_size:
                # 先保存当前chunk（如果有内容）
                if current_chunk.strip():
                    chunk = self._create_chunk(
                        current_chunk.strip(), 
                        metadata, 
                        file_path, 
                        chunk_index, 
                        current_start, 
                        current_start + len(current_chunk)
                    )
                    chunks.append(chunk)
                    chunk_index += 1
                
                # 对超长段落进行强制分割
                long_chunks = self._force_split_paragraph(paragraph, metadata, file_path, chunk_index, current_start)
                chunks.extend(long_chunks)
                chunk_index += len(long_chunks)
                
                current_chunk = ""
                current_start += len(paragraph)
                
            # 如果添加这个段落会超过chunk_size
            elif len(current_chunk) + len(paragraph) > self.chunk_size:
                # 保存当前chunk
                if current_chunk.strip():
                    chunk = self._create_chunk(
                        current_chunk.strip(), 
                        metadata, 
                        file_path, 
                        chunk_index, 
                        current_start, 
                        current_start + len(current_chunk)
                    )
                    chunks.append(chunk)
                    chunk_index += 1
                
                # 开始新的chunk，考虑重叠
                overlap_text = self._get_overlap_text(current_chunk, self.chunk_overlap)
                current_chunk = overlap_text + paragraph
                current_start = current_start + len(current_chunk) - len(overlap_text) - len(paragraph)
                
            else:
                # 添加到当前chunk
                current_chunk += paragraph
        
        # 处理最后一个chunk
        if current_chunk.strip():
            chunk = self._create_chunk(
                current_chunk.strip(), 
                metadata, 
                file_path, 
                chunk_index, 
                current_start, 
                current_start + len(current_chunk)
            )
            chunks.append(chunk)
        
        return chunks
    
    def _split_by_structure(self, text: str) -> List[str]:
        """
        按Markdown结构分割文本
        
        Args:
            text: 文本内容
            
        Returns:
            段落列表
        """
        # 按双换行符分割段落，保持结构
        paragraphs = re.split(r'\n\s*\n', text)
        
        # 确保每个段落都以换行符结尾（除了最后一个）
        result = []
        for i, paragraph in enumerate(paragraphs):
            if paragraph.strip():
                if i < len(paragraphs) - 1:
                    result.append(paragraph + '\n\n')
                else:
                    result.append(paragraph)
        
        return result
    
    def _force_split_paragraph(self, paragraph: str, metadata: Dict[str, Any], 
                             file_path: str, start_chunk_index: int, start_pos: int) -> List[DocumentChunk]:
        """
        强制分割超长段落
        
        Args:
            paragraph: 段落内容
            metadata: 元数据
            file_path: 文件路径
            start_chunk_index: 起始chunk索引
            start_pos: 起始位置
            
        Returns:
            分块列表
        """
        chunks = []
        chunk_index = start_chunk_index
        pos = start_pos
        
        # 按句子分割（简单实现）
        sentences = re.split(r'[。！？.!?]\s*', paragraph)
        
        current_chunk = ""
        for sentence in sentences:
            if not sentence.strip():
                continue
                
            sentence = sentence + '。'  # 恢复句号
            
            if len(current_chunk) + len(sentence) > self.chunk_size and current_chunk:
                # 保存当前chunk
                chunk = self._create_chunk(current_chunk.strip(), metadata, file_path, chunk_index, pos, pos + len(current_chunk))
                chunks.append(chunk)
                chunk_index += 1
                pos += len(current_chunk)
                
                # 开始新chunk
                current_chunk = sentence
            else:
                current_chunk += sentence
        
        # 处理最后一个chunk
        if current_chunk.strip():
            chunk = self._create_chunk(current_chunk.strip(), metadata, file_path, chunk_index, pos, pos + len(current_chunk))
            chunks.append(chunk)
        
        return chunks
    
    def _get_overlap_text(self, text: str, overlap_size: int) -> str:
        """
        获取重叠文本
        
        Args:
            text: 原文本
            overlap_size: 重叠大小
            
        Returns:
            重叠文本
        """
        if len(text) <= overlap_size:
            return text
        
        # 尝试在句子边界处截取
        overlap_text = text[-overlap_size:]
        sentence_start = overlap_text.find('。')
        
        if sentence_start != -1:
            return overlap_text[sentence_start + 1:]
        
        return overlap_text
    
    def _create_chunk(self, content: str, metadata: Dict[str, Any], 
                     file_path: str, chunk_index: int, start_pos: int, end_pos: int) -> DocumentChunk:
        """
        创建文档分块
        
        Args:
            content: 分块内容
            metadata: 文档元数据
            file_path: 文件路径
            chunk_index: 分块索引
            start_pos: 开始位置
            end_pos: 结束位置
            
        Returns:
            文档分块对象
        """
        # 生成唯一ID
        chunk_id = hashlib.md5(f"{file_path}_{chunk_index}_{content[:50]}".encode()).hexdigest()[:16]
        
        # 分块元数据
        chunk_metadata = {
            **metadata,
            'chunk_length': len(content),
            'chunk_index': chunk_index,
            'start_pos': start_pos,
            'end_pos': end_pos,
            'created_at': datetime.now()
        }
        
        return DocumentChunk(
            id=chunk_id,
            content=content,
            metadata=chunk_metadata,
            source_file=file_path,
            chunk_index=chunk_index,
            start_pos=start_pos,
            end_pos=end_pos
        )

# 为了向后兼容，保留原类名
DocumentProcessor = MarkdownDocumentProcessor