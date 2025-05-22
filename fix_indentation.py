import re

def fix_file():
    # 读取文件
    with open('knowledge/self_organizing_knowledge.py', 'r', encoding='utf-8') as f:
        content = f.readlines()
    
    # 修复缩进问题
    for i in range(len(content)):
        # 修复 clusters = self._cluster_concepts() 缩进问题
        if '        clusters = self._cluster_concepts()' in content[i]:
            content[i] = content[i].replace('        clusters', '            clusters')
        
        # 修复 relation_changes = self._optimize_relations() 缩进问题
        if '        relation_changes = self._optimize_relations()' in content[i]:
            content[i] = content[i].replace('        relation_changes', '            relation_changes')
        
        # 修复 hierarchy_changes = self._update_hierarchies(clusters) 缩进问题
        if '        hierarchy_changes = self._update_hierarchies(clusters)' in content[i]:
            content[i] = content[i].replace('        hierarchy_changes', '            hierarchy_changes')
        
        # 修复 for target_id, relation_type, confidence in relations: 缩进问题
        if '            for target_id, relation_type, confidence in relations:' in content[i] and '                for' not in content[i-1]:
            content[i] = content[i].replace('            for', '                for')
            
        # 修复 if source_id not in self.concepts: 缩进问题
        if '            if source_id not in self.concepts:' in content[i] and '                if' not in content[i-1]:
            content[i] = content[i].replace('            if', '                if')
    
    # 写回文件
    with open('knowledge/self_organizing_knowledge.py', 'w', encoding='utf-8') as f:
        f.writelines(content)
    
    print("缩进问题修复完成！")

if __name__ == "__main__":
    fix_file() 