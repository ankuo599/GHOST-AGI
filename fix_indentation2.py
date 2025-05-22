def fix_file():
    try:
        # 读取文件内容
        with open('knowledge/self_organizing_knowledge.py', 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 查找并修复具体缩进问题
        content = content.replace('try:\n            # 阶段1: 概念聚类\n            self.logger.info("执行概念聚类...")\n        clusters = self._cluster_concepts()',
                               'try:\n            # 阶段1: 概念聚类\n            self.logger.info("执行概念聚类...")\n            clusters = self._cluster_concepts()')
        
        content = content.replace('self.logger.info("优化概念关系...")\n        relation_changes = self._optimize_relations()',
                               'self.logger.info("优化概念关系...")\n            relation_changes = self._optimize_relations()')
        
        content = content.replace('self.logger.info("更新层次结构...")\n        hierarchy_changes = self._update_hierarchies(clusters)',
                               'self.logger.info("更新层次结构...")\n            hierarchy_changes = self._update_hierarchies(clusters)')
        
        content = content.replace('            for source_id, relations in self.relations.items():\n            for target_id',
                               '            for source_id, relations in self.relations.items():\n                for target_id')
        
        content = content.replace('            if concept_id in self.concepts:\n                result["warnings"].append',
                               '                if concept_id in self.concepts:\n                    result["warnings"].append')
        
        content = content.replace('if source_id not in self.concepts:\n                    result["warnings"]',
                               'if source_id not in self.concepts:\n                        result["warnings"]')
        
        # 写回文件
        with open('knowledge/self_organizing_knowledge.py', 'w', encoding='utf-8') as f:
            f.write(content)
        
        print("缩进问题修复成功！")
    
    except Exception as e:
        print(f"修复过程发生错误: {e}")

if __name__ == '__main__':
    fix_file() 