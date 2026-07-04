#!/usr/bin/env python3
"""
专门检查OpenFly-rlds数据集的脚本
"""

import tensorflow as tf
import os
import glob
import sys
from tqdm import tqdm


def check_openfly_dataset(base_path="/mnt/sdc/weiguanzhao/OpenFly-rlds-my/vln_history/1.0.0/"):
    """检查OpenFly数据集"""

    # 查找所有训练文件
    pattern = os.path.join(base_path, "vln_history-train.tfrecord-*-of-01024")
    files = sorted(glob.glob(pattern))

    if not files:
        print(f"错误: 没有找到匹配 {pattern} 的文件")
        return

    print(f"找到 {len(files)} 个训练文件")
    print(f"文件范围: {os.path.basename(files[0])} 到 {os.path.basename(files[-1])}")

    corrupted_files = []
    healthy_files = []

    # 检查每个文件
    for filepath in tqdm(files, desc="检查文件"):
        try:
            # 检查文件是否存在和大小
            if not os.path.exists(filepath):
                corrupted_files.append((filepath, "文件不存在"))
                continue

            file_size = os.path.getsize(filepath)
            if file_size == 0:
                corrupted_files.append((filepath, "文件大小为0"))
                continue

            # 使用TensorFlow 2.x方法检查
            try:
                dataset = tf.data.TFRecordDataset(filepath)

                # 只检查前5条记录来快速判断
                record_count = 0
                for record in dataset.take(5):
                    record_count += 1

                if record_count > 0:
                    # 如果前5条正常，再检查总记录数
                    total_records = 0
                    for record in dataset:
                        total_records += 1

                    healthy_files.append((filepath, total_records))
                else:
                    corrupted_files.append((filepath, "无法读取任何记录"))

            except tf.errors.DataLossError as e:
                corrupted_files.append((filepath, f"数据损坏: {e}"))
            except Exception as e:
                corrupted_files.append((filepath, f"TF读取错误: {type(e).__name__}"))

        except Exception as e:
            corrupted_files.append((filepath, f"检查错误: {type(e).__name__}"))

    # 输出结果
    print(f"\n检查完成:")
    print(f"  总文件数: {len(files)}")
    print(f"  健康文件数: {len(healthy_files)}")
    print(f"  损坏文件数: {len(corrupted_files)}")

    if healthy_files:
        total_records = sum(count for _, count in healthy_files)
        avg_records = total_records / len(healthy_files) if healthy_files else 0
        print(f"  健康文件总记录数: {total_records}")
        print(f"  平均每个文件记录数: {avg_records:.1f}")

    if corrupted_files:
        print(f"\n损坏文件列表:")
        for filepath, error in corrupted_files:
            filename = os.path.basename(filepath)
            file_num = filename.split('-')[-1].split('.')[0]  # 提取文件编号
            print(f"  {file_num}: {filename} - {error}")

        # 生成修复脚本
        generate_repair_script(corrupted_files)

    return healthy_files, corrupted_files


def generate_repair_script(corrupted_files):
    """生成修复脚本"""
    if not corrupted_files:
        return

    script_content = """#!/bin/bash
# 修复脚本 - 删除损坏的TFRecord文件

echo "以下损坏文件将被删除:"
"""

    for filepath, error in corrupted_files:
        script_content += f"echo \"  {os.path.basename(filepath)} - {error}\"\n"

    script_content += "\nread -p \"是否继续删除? (y/n): \" -n 1 -r\necho\n"
    script_content += "if [[ $REPLY =~ ^[Yy]$ ]]\nthen\n"

    for filepath, _ in corrupted_files:
        script_content += f"    rm -v \"{filepath}\"\n"

    script_content += "    echo \"删除完成\"\nelse\n    echo \"取消删除\"\nfi\n"

    script_file = "repair_corrupted_files.sh"
    with open(script_file, 'w') as f:
        f.write(script_content)

    print(f"\n已生成修复脚本: {script_file}")
    print(f"运行命令: bash {script_file}")

    # 也生成Python修复脚本
    python_script = """#!/usr/bin/env python3
import os
import tensorflow as tf

def repair_with_skip_errors(corrupted_files):
    \"\"\"尝试修复损坏文件（跳过错误记录）\"\"\"
    for filepath in corrupted_files:
        try:
            output_file = filepath.replace('.tfrecord', '_repaired.tfrecord')
            print(f"修复: {os.path.basename(filepath)} -> {os.path.basename(output_file)}")

            # 使用ignore_errors跳过损坏记录
            dataset = tf.data.TFRecordDataset(filepath)
            dataset = dataset.apply(tf.data.experimental.ignore_errors())

            writer = tf.io.TFRecordWriter(output_file)
            recovered = 0

            for record in dataset:
                writer.write(record.numpy())
                recovered += 1

            writer.close()
            print(f"  成功恢复 {recovered} 条记录")

        except Exception as e:
            print(f"  修复失败: {e}")

# 要修复的文件列表
corrupted_files = [
"""

    for filepath, _ in corrupted_files:
        python_script += f"    '{filepath}',\n"

    python_script += "]\n\nif __name__ == '__main__':\n    repair_with_skip_errors(corrupted_files)\n"

    python_script_file = "repair_tfrecords.py"
    with open(python_script_file, 'w') as f:
        f.write(python_script)

    print(f"已生成Python修复脚本: {python_script_file}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        base_path = sys.argv[1]
    else:
        base_path = "/mnt/sdc/weiguanzhao/OpenFly-rlds-my/vln_history/1.0.0/"

    healthy, corrupted = check_openfly_dataset(base_path)