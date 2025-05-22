def fix_specific_lines():
    try:
        # 读取文件
        with open('knowledge/self_organizing_knowledge.py', 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        # 修复 line 448 - clusters = self._cluster_concepts() 的缩进
        if len(lines) >= 448 and 'clusters = self._cluster_concepts()' in lines[447]:
            lines[447] = '            clusters = self._cluster_concepts()\n'
        
        # 修复 line 460 - relation_changes = self._optimize_relations() 的缩进
        if len(lines) >= 460 and 'relation_changes = self._optimize_relations()' in lines[459]:
            lines[459] = '            relation_changes = self._optimize_relations()\n'
        
        # 修复 line 474 - hierarchy_changes = self._update_hierarchies(clusters) 的缩进
        if len(lines) >= 474 and 'hierarchy_changes = self._update_hierarchies(clusters)' in lines[473]:
            lines[473] = '            hierarchy_changes = self._update_hierarchies(clusters)\n'
        
        # 修复 line 733-734 的缩进
        if len(lines) >= 734 and 'for target_id, relation_type, confidence in relations' in lines[732]:
            lines[732] = '                for target_id, relation_type, confidence in relations:\n'
            if 'knowledge_graph["relations"].append' in lines[733]:
                lines[733] = '                    knowledge_graph["relations"].append({\n'
        
        # 修复 line 815-823 的缩进问题
        if len(lines) >= 815 and 'if concept_id in self.concepts' in lines[814]:
            lines[814] = '                if concept_id in self.concepts:\n'
            if 'result["warnings"].append' in lines[815]:
                lines[815] = '                    result["warnings"].append(f"概念ID已存在，将更新: {concept_id}")\n'
            if '# 更新概念' in lines[816]:
                lines[816] = '                    # 更新概念\n'
            if 'self.concepts[concept_id].update' in lines[817]:
                lines[817] = '                    self.concepts[concept_id].update(concept_data)\n'
            if 'else:' in lines[818]:
                lines[818] = '                else:\n'
            if '# 添加概念' in lines[819]:
                lines[819] = '                    # 添加概念\n'
            if 'concept_data["id"] = concept_id' in lines[820]:
                lines[820] = '                    concept_data["id"] = concept_id\n'
            if 'self.add_concept(concept_data)' in lines[821]:
                lines[821] = '                    self.add_concept(concept_data)\n'
            if 'result["imported_concepts"] += 1' in lines[822]:
                lines[822] = '                    result["imported_concepts"] += 1\n'
        
        # 修复 line 842-843 的缩进问题
        if len(lines) >= 843 and 'if source_id not in self.concepts' in lines[841]:
            lines[841] = '                if source_id not in self.concepts:\n'
            if 'result["warnings"].append' in lines[842]:
                lines[842] = '                    result["warnings"].append(f"关系源概念不存在，跳过: {source_id}")\n'
            if 'continue' in lines[843]:
                lines[843] = '                    continue\n'
        
        # 修复 line 920 的缩进
        if len(lines) >= 920 and 'similarity = self._calculate_similarity' in lines[919]:
            lines[919] = '                similarity = self._calculate_similarity(concept_vector, other_vector)\n'
        
        # 修复 line 1010 的缩进
        if len(lines) >= 1010 and 'self.reorganize_knowledge()' in lines[1009]:
            lines[1009] = '                self.reorganize_knowledge()\n'
        
        # 写回文件
        with open('knowledge/self_organizing_knowledge.py', 'w', encoding='utf-8') as f:
            f.writelines(lines)
        
        print("特定行缩进问题修复完成！")
        
    except Exception as e:
        print(f"修复过程发生错误: {e}")

if __name__ == "__main__":
    fix_specific_lines() 