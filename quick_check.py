#!/usr/bin/env python3
"""
快速TFRecord检查工具 - 修复TensorFlow 2.x兼容性问题
"""

import tensorflow as tf
import glob
import sys
import os


def check_file_with_tf2(filepath):
    """使用TensorFlow 2.x方法检查文件"""
    try:
        # 使用tf.data.TFRecordDataset
        dataset = tf.data.TFRecordDataset(filepath)

        # 尝试读取前10条记录
        record_count = 0
        for record in dataset.take(10000):
            record_count += 1

        if record_count > 0:
            return True, record_count, None
        else:
            return False, 0, "文件为空"

    except tf.errors.DataLossError as e:
        return False, 0, f"数据损坏: {e}"
    except Exception as e:
        return False, 0, f"其他错误: {type(e).__name__}: {str(e)}"


def quick_check_tfrecords_tf2(file_pattern):
    """快速检查一批TFRecord文件 - TensorFlow 2.x版本"""
    files = glob.glob(file_pattern)
    if not files:
        print(f"没有找到匹配 {file_pattern} 的文件")
        return

    print(f"检查 {len(files)} 个文件...")

    corrupted_files = []
    healthy_files = []

    for filepath in files:
        # 首先检查文件是否存在且大小不为0
        if not os.path.exists(filepath):
            corrupted_files.append((filepath, "文件不存在"))
            print(f"✗ {filepath}: 文件不存在")
            continue

        if os.path.getsize(filepath) == 0:
            corrupted_files.append((filepath, "文件大小为0"))
            print(f"✗ {filepath}: 文件大小为0")
            continue

        # 使用TensorFlow检查
        success, count, error = check_file_with_tf2(filepath)

        if success:
            healthy_files.append((filepath, count))
            print(f"✓ {os.path.basename(filepath)}: 正常 ({count}条记录)")
        else:
            corrupted_files.append((filepath, error))
            print(f"✗ {os.path.basename(filepath)}: {error}")

    # 总结
    print(f"\n总结:")
    print(f"  总文件数: {len(files)}")
    print(f"  健康文件数: {len(healthy_files)}")
    print(f"  损坏文件数: {len(corrupted_files)}")

    if healthy_files:
        total_records = sum(count for _, count in healthy_files)
        print(f"  健康文件总记录数: {total_records}")

    if corrupted_files:
        print(f"\n损坏文件列表 (前10个):")
        for filepath, error in corrupted_files[:10]:
            print(f"  - {os.path.basename(filepath)}: {error}")

        if len(corrupted_files) > 10:
            print(f"  ... 还有 {len(corrupted_files) - 10} 个损坏文件")

        return False
    return True


def check_specific_files_tf2(file_list):
    """检查指定的文件列表"""
    with open(file_list, 'r') as f:
        files = [line.strip() for line in f if line.strip()]

    return quick_check_tfrecords_tf2(files)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python quick_check_tf2.py /path/to/tfrecords/*.tfrecord")
        print("或: python quick_check_tf2.py --list file_list.txt")
        sys.exit(1)

    if sys.argv[1] == "--list" and len(sys.argv) >= 3:
        success = check_specific_files_tf2(sys.argv[2])
    else:
        success = quick_check_tfrecords_tf2(sys.argv[1])

    sys.exit(0 if success else 1)