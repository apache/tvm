# 从测试失败中提取缺失的操作符
import re

test_output = """<用户粘贴的测试输出>"""

# 提取所有 Unixsupported function types 的错误
missing_ops = set()
for line in test_output.split('\n'):
    if 'Unsupported function types' in line:
        # 提取操作符名称
        match = re.search(r"Unsupported function types \[(.*?)\]", line)
        if match:
            ops_str = match.group(1)
            # 移除引号并分割
            ops = [op.strip().strip("'\"") for op in ops_str.split(',')]
            missing_ops.update(ops)

# 排序输出
print("缺失的操作符:")
for op in sorted(missing_ops):
    print(f"  - {op}")

print(f"\n总共缺失 {len(missing_ops)} 个操作符")
