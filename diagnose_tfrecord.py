#!/usr/bin/env python3
"""
诊断TFRecord文件损坏原因
"""

import os
import sys
import struct
import hashlib


def analyze_file(filepath):
    """详细分析文件内容"""
    print(f"\n分析文件: {os.path.basename(filepath)}")
    print("=" * 60)

    # 1. 检查文件是否存在和大小
    if not os.path.exists(filepath):
        print("❌ 文件不存在")
        return

    file_size = os.path.getsize(filepath)
    print(f"文件大小: {file_size} 字节 ({file_size / 1024 / 1024:.2f} MB)")

    if file_size == 0:
        print("❌ 文件为空 (0字节)")
        return

    # 2. 检查文件头部
    with open(filepath, 'rb') as f:
        # 读取前100字节
        header = f.read(100)

        if len(header) < 12:
            print(f"❌ 文件太小，无法分析 (只有{len(header)}字节)")
            return

        print(f"前100字节 (十六进制):")
        for i in range(0, min(100, len(header)), 16):
            hex_str = ' '.join(f'{b:02x}' for b in header[i:i + 16])
            ascii_str = ''.join(chr(b) if 32 <= b < 127 else '.' for b in header[i:i + 16])
            print(f"  {i:04x}: {hex_str:<48} {ascii_str}")

        # 3. 尝试解析TFRecord格式
        print(f"\nTFRecord格式分析:")

        # TFRecord格式：每个记录包含：
        # 1. 8字节的CRC32校验码 (用于长度)
        # 2. 4字节的数据长度 (little-endian)
        # 3. 数据内容
        # 4. 4字节的CRC32校验码 (用于数据)

        # 检查前12字节（长度+CRC）
        if len(header) >= 12:
            # 获取数据长度（4字节，little-endian）
            data_length = struct.unpack('<I', header[8:12])[0]
            print(f"  数据长度字段: {data_length} 字节")

            # 检查长度是否合理
            total_record_size = 8 + 4 + data_length + 4  # CRC(8) + 长度(4) + 数据 + CRC(4)

            if data_length == 0:
                print("  ⚠️  数据长度为0 - 可能是空记录")
            elif data_length > 100 * 1024 * 1024:  # 超过100MB
                print(f"  ❌ 数据长度异常大 ({data_length}字节)")
            elif total_record_size > file_size:
                print(f"  ❌ 记录大小({total_record_size}) > 文件大小({file_size})")
            else:
                print(f"  ✓ 数据长度看起来合理")

        # 4. 检查文件类型（通过魔术数字）
        magic_numbers = {
            b'\x1f\x8b': 'GZIP压缩文件',
            b'PK\x03\x04': 'ZIP文件',
            b'\x89PNG\r\n\x1a\n': 'PNG图像',
            b'\xff\xd8\xff': 'JPEG图像',
            b'%PDF': 'PDF文件',
            b'TFRecord': 'TFRecord文件',
            b'PAR1': 'Parquet文件',
        }

        print(f"\n文件类型检测:")
        for magic, filetype in magic_numbers.items():
            if header.startswith(magic):
                print(f"  ✓ 检测为: {filetype}")
                break
        else:
            # 检查是否为文本文件
            text_chars = sum(1 for b in header[:100] if 32 <= b < 127 or b in (9, 10, 13))
            if text_chars > 80:
                print("  ⚠️  可能是文本文件")
                # 尝试解码为UTF-8
                try:
                    text_content = header[:100].decode('utf-8', errors='ignore')
                    print(f"  文本内容: {text_content[:50]}...")
                except:
                    pass

        # 5. 检查文件哈希
        f.seek(0)
        file_hash = hashlib.md5(f.read()).hexdigest()
        print(f"\n文件MD5哈希: {file_hash}")

        # 6. 与其他正常文件比较
        print(f"\n建议:")
        if file_size == 0:
            print("  1. 删除此空文件")
            print("  2. 重新生成数据")
        elif data_length == 0 and header[8:12] == b'\x00\x00\x00\x00':
            print("  1. 文件包含空记录，可能是生成错误")
            print("  2. 删除并重新生成")
        else:
            print("  1. 文件可能是其他格式，不是TFRecord")
            print("  2. 检查数据生成脚本")


def compare_with_good_file(bad_file, good_file_pattern):
    """与正常文件对比"""
    import glob

    good_files = glob.glob(good_file_pattern)
    if not good_files:
        print("找不到正常文件进行对比")
        return

    good_file = good_files[0]

    print(f"\n对比: {os.path.basename(bad_file)} vs {os.path.basename(good_file)}")

    bad_size = os.path.getsize(bad_file)
    good_size = os.path.getsize(good_file)

    print(f"大小: {bad_size} vs {good_size} (差值: {bad_size - good_size})")

    # 比较头部
    with open(bad_file, 'rb') as f1, open(good_file, 'rb') as f2:
        bad_header = f1.read(100)
        good_header = f2.read(100)

        for i in range(0, 100, 16):
            bad_hex = ' '.join(f'{b:02x}' for b in bad_header[i:i + 16])
            good_hex = ' '.join(f'{b:02x}' for b in good_header[i:i + 16])

            if bad_hex != good_hex:
                print(f"  字节 {i:04x}:")
                print(f"    坏文件: {bad_hex}")
                print(f"    好文件: {good_hex}")
                print(f"    差异: ", end="")

                for j, (b1, b2) in enumerate(zip(bad_header[i:i + 16], good_header[i:i + 16])):
                    if b1 != b2:
                        print(f"[{j}:{b1:02x}!={b2:02x}] ", end="")
                print()
                break
        else:
            print("  前100字节完全相同")


def check_directory_for_empty_files(directory):
    """检查目录中的空文件"""
    import glob

    pattern = os.path.join(directory, "*.tfrecord*")
    files = glob.glob(pattern)

    empty_files = []
    small_files = []

    for filepath in files:
        size = os.path.getsize(filepath)
        if size == 0:
            empty_files.append(filepath)
        elif size < 1024:  # 小于1KB
            small_files.append(filepath)

    print(f"\n目录 {directory} 检查结果:")
    print(f"  总文件数: {len(files)}")
    print(f"  空文件数 (0字节): {len(empty_files)}")
    print(f"  小文件数 (<1KB): {len(small_files)}")

    if empty_files:
        print(f"\n空文件列表:")
        for f in empty_files[:10]:  # 只显示前10个
            print(f"  - {os.path.basename(f)}")

    if small_files:
        print(f"\n小文件列表:")
        for f in small_files[:10]:
            size = os.path.getsize(f)
            print(f"  - {os.path.basename(f)}: {size} 字节")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法:")
        print("  1. 分析单个文件: python diagnose_tfrecord.py 文件名")
        print("  2. 检查目录: python diagnose_tfrecord.py --dir 目录")
        print("  3. 对比文件: python diagnose_tfrecord.py --compare 坏文件 '好文件模式'")
        sys.exit(1)

    if sys.argv[1] == "--dir":
        check_directory_for_empty_files(sys.argv[2])
    elif sys.argv[1] == "--compare":
        compare_with_good_file(sys.argv[2], sys.argv[3])
    else:
        analyze_file(sys.argv[1])