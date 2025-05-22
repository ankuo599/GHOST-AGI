"""
知识库模块
实现知识的存储、检索和更新
"""

from typing import Dict, List, Any, Optional, Set, Tuple
import logging
from dataclasses import dataclass
import json
import time
import numpy as np
from pathlib import Path
import sqlite3
from datetime import datetime
import hashlib

@dataclass
class Knowledge:
    """知识条目"""
    id: str
    content: str
    type: str  # 知识类型：fact, rule, concept, relation
    domain: str  # 所属领域
    confidence: float  # 置信度
    source: str  # 知识来源
    timestamp: float  # 创建时间
    last_updated: float  # 最后更新时间
    metadata: Dict[str, Any]  # 元数据
    embeddings: Optional[np.ndarray] = None  # 语义向量

class KnowledgeBase:
    """知识库"""
    def __init__(self, db_path: str = "knowledge.db"):
        self.db_path = db_path
        self.logger = logging.getLogger("KnowledgeBase")
        self._init_db()
        
    def _init_db(self):
        """初始化数据库"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # 创建知识表
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS knowledge (
                    id TEXT PRIMARY KEY,
                    content TEXT NOT NULL,
                    type TEXT NOT NULL,
                    domain TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    source TEXT NOT NULL,
                    timestamp REAL NOT NULL,
                    last_updated REAL NOT NULL,
                    metadata TEXT NOT NULL,
                    embeddings BLOB
                )
            """)
            
            # 创建索引
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_domain ON knowledge(domain)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_type ON knowledge(type)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON knowledge(timestamp)")
            
            conn.commit()
            
    def add_knowledge(self, knowledge: Knowledge) -> bool:
        """添加知识"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # 序列化元数据和向量
                metadata_json = json.dumps(knowledge.metadata)
                embeddings_bytes = knowledge.embeddings.tobytes() if knowledge.embeddings is not None else None
                
                cursor.execute("""
                    INSERT OR REPLACE INTO knowledge
                    (id, content, type, domain, confidence, source, timestamp, last_updated, metadata, embeddings)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    knowledge.id,
                    knowledge.content,
                    knowledge.type,
                    knowledge.domain,
                    knowledge.confidence,
                    knowledge.source,
                    knowledge.timestamp,
                    knowledge.last_updated,
                    metadata_json,
                    embeddings_bytes
                ))
                
                conn.commit()
        return True
        
        except Exception as e:
            self.logger.error(f"添加知识失败: {str(e)}")
            return False
            
    def get_knowledge(self, knowledge_id: str) -> Optional[Knowledge]:
        """获取知识"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT * FROM knowledge WHERE id = ?", (knowledge_id,))
                row = cursor.fetchone()
                
                if row:
                    return self._row_to_knowledge(row)
                return None
                
        except Exception as e:
            self.logger.error(f"获取知识失败: {str(e)}")
            return None
            
    def search_knowledge(self, 
                        query: str,
                        domain: Optional[str] = None,
                        type: Optional[str] = None,
                        min_confidence: float = 0.0,
                        limit: int = 100) -> List[Knowledge]:
        """搜索知识"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # 构建查询条件
                conditions = ["content LIKE ?"]
                params = [f"%{query}%"]
                
                if domain:
                    conditions.append("domain = ?")
                    params.append(domain)
                    
                if type:
                    conditions.append("type = ?")
                    params.append(type)
                    
                conditions.append("confidence >= ?")
                params.append(min_confidence)
                
                # 执行查询
                query = f"""
                    SELECT * FROM knowledge
                    WHERE {" AND ".join(conditions)}
                    ORDER BY confidence DESC, timestamp DESC
                    LIMIT ?
                """
                params.append(limit)
                
                cursor.execute(query, params)
                rows = cursor.fetchall()
                
                return [self._row_to_knowledge(row) for row in rows]
                
        except Exception as e:
            self.logger.error(f"搜索知识失败: {str(e)}")
            return []
            
    def update_knowledge(self, knowledge_id: str, updates: Dict[str, Any]) -> bool:
        """更新知识"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # 获取现有知识
                cursor.execute("SELECT * FROM knowledge WHERE id = ?", (knowledge_id,))
                row = cursor.fetchone()
                
                if not row:
                    return False
                    
                # 更新字段
                knowledge = self._row_to_knowledge(row)
                for key, value in updates.items():
                    if hasattr(knowledge, key):
                        setattr(knowledge, key, value)
                        
                knowledge.last_updated = time.time()
                
                # 保存更新
                return self.add_knowledge(knowledge)
                
        except Exception as e:
            self.logger.error(f"更新知识失败: {str(e)}")
            return False
            
    def delete_knowledge(self, knowledge_id: str) -> bool:
        """删除知识"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("DELETE FROM knowledge WHERE id = ?", (knowledge_id,))
                conn.commit()
                return cursor.rowcount > 0
                
        except Exception as e:
            self.logger.error(f"删除知识失败: {str(e)}")
            return False
            
    def get_domain_stats(self) -> Dict[str, Any]:
        """获取领域统计信息"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # 获取领域统计
                cursor.execute("""
                    SELECT domain, COUNT(*) as count, AVG(confidence) as avg_confidence
                    FROM knowledge
                    GROUP BY domain
                """)
                domain_stats = {
                    row[0]: {
                        "count": row[1],
                        "avg_confidence": row[2]
                    }
                    for row in cursor.fetchall()
                }
                
                # 获取类型统计
                cursor.execute("""
                    SELECT type, COUNT(*) as count
                    FROM knowledge
                    GROUP BY type
                """)
                type_stats = {
                    row[0]: row[1]
                    for row in cursor.fetchall()
                }
                
                return {
                    "total_knowledge": sum(stat["count"] for stat in domain_stats.values()),
                    "domains": domain_stats,
                    "types": type_stats,
                    "last_updated": time.time()
                }
                
        except Exception as e:
            self.logger.error(f"获取统计信息失败: {str(e)}")
            return {}
            
    def _row_to_knowledge(self, row: Tuple) -> Knowledge:
        """将数据库行转换为知识对象"""
        # 反序列化元数据和向量
        metadata = json.loads(row[8])
        embeddings = np.frombuffer(row[9]) if row[9] else None
        
        return Knowledge(
            id=row[0],
            content=row[1],
            type=row[2],
            domain=row[3],
            confidence=row[4],
            source=row[5],
            timestamp=row[6],
            last_updated=row[7],
            metadata=metadata,
            embeddings=embeddings
        )
        
    def export_knowledge(self, file_path: str) -> bool:
        """导出知识库"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT * FROM knowledge")
                rows = cursor.fetchall()
                
                knowledge_list = [self._row_to_knowledge(row) for row in rows]
                
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(
                        [{
                            "id": k.id,
                            "content": k.content,
                            "type": k.type,
                            "domain": k.domain,
                            "confidence": k.confidence,
                            "source": k.source,
                            "timestamp": k.timestamp,
                            "last_updated": k.last_updated,
                            "metadata": k.metadata
                        } for k in knowledge_list],
                        f,
                        ensure_ascii=False,
                        indent=2
                    )
                    
                return True
                
        except Exception as e:
            self.logger.error(f"导出知识库失败: {str(e)}")
            return False
            
    def import_knowledge(self, file_path: str) -> bool:
        """导入知识库"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                knowledge_list = json.load(f)
                
            success = True
            for k in knowledge_list:
                knowledge = Knowledge(
                    id=k["id"],
                    content=k["content"],
                    type=k["type"],
                    domain=k["domain"],
                    confidence=k["confidence"],
                    source=k["source"],
                    timestamp=k["timestamp"],
                    last_updated=k["last_updated"],
                    metadata=k["metadata"]
                )
                
                if not self.add_knowledge(knowledge):
                    success = False
                    
            return success
            
        except Exception as e:
            self.logger.error(f"导入知识库失败: {str(e)}")
            return False
            
    def merge_knowledge(self, other_kb: 'KnowledgeBase') -> bool:
        """合并知识库"""
        try:
            with sqlite3.connect(other_kb.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT * FROM knowledge")
                rows = cursor.fetchall()
                
                success = True
                for row in rows:
                    knowledge = other_kb._row_to_knowledge(row)
                    if not self.add_knowledge(knowledge):
                        success = False
                        
                return success
                
        except Exception as e:
            self.logger.error(f"合并知识库失败: {str(e)}")
            return False
            
    def cleanup(self, max_age_days: int = 30) -> int:
        """清理过期知识"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # 删除过期知识
                cutoff_time = time.time() - (max_age_days * 24 * 3600)
                cursor.execute(
                    "DELETE FROM knowledge WHERE last_updated < ?",
                    (cutoff_time,)
                )
                
                deleted_count = cursor.rowcount
                conn.commit()
                return deleted_count
                
        except Exception as e:
            self.logger.error(f"清理知识库失败: {str(e)}")
            return 0 