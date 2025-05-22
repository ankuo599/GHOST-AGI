
# 这是一个包含多种代码问题的测试类
class BadImplementation:
    def __init__(self, maxItems=100):
        self.items = []
        self.maxItems = maxItems
        self.counter = 0
    
    # 命名不规范，缺少文档
    def add_item(self, item):
        # 缺少错误处理
        self.items.append(item)
        self.counter = self.counter + 1
        # 性能问题 - 每次都完整遍历列表
        if len([x for x in self.items if x is not None]) > self.maxItems:
            # 内存问题 - 不必要的列表副本
            self.items = list(self.items)
            self.items.pop(0)
    
    # 重复代码片段示例
    def get_items_starting_with(self, prefix):
        result = []
        for item in self.items:
            if str(item).startswith(prefix):
                result.append(item)
        return result
    
    # 几乎相同的代码，应该合并
    def get_items_ending_with(self, suffix):
        result = []
        for item in self.items:
            if str(item).endswith(suffix):
                result.append(item)
        return result
        
    # TODO: 考虑将此复杂函数拆分为多个更小的函数以减少复杂度
    def process_data(self):
        # 大量嵌套的条件判断，复杂度高
        processed = []
        for i, item in enumerate(self.items):
            if item is not None:
                if isinstance(item, str):
                    if len(item) > 0:
                        if item[0].isalpha():
                            if item not in processed:
                                processed.append(item.upper())
                        else:
                            if len(item) > 1:
                                processed.append(item[1:])
                            else:
                                processed.append("")
                    else:
                        processed.append("")
                elif isinstance(item, int):
                    if item > 0:
                        if item % 2 == 0:
                            processed.append(str(item * 2))
                        else:
                            processed.append(str(item * 3))
                    else:
                        processed.append("0")
                else:
                    processed.append(str(item))
            else:
                processed.append("")
        return processed