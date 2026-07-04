
import numpy as np

# 假设的数据 - 请替换为您的实际数据
# 每个特征有10个值
sr = [16.23, 19.83, 17.10, 20.32, 18.34]      # 特征sr的10个值
# sr = [16.23, 16.48, 19.83, 17.10, 20.32, 18.34]      # 特征sr的10个值
# sr = [16.23, 16.48, 19.83, 17.10, 20.32, 16.48, 18.34]      # 特征sr的10个值
# sr = [16.23, 14.50, 16.48, 19.83, 17.10, 13.14, 20.32, 16.85, 16.48, 18.34]      # 特征sr的10个值
osr = [37.92, 36.69, 41.39, 38.41, 44.49]     # 特征osr的10个值
# osr = [37.92, 45.48, 36.69, 41.39, 38.41, 44.49]     # 特征osr的10个值
# osr = [37.92, 45.48, 36.69, 41.39, 38.41, 48.70, 44.49]     # 特征osr的10个值
# osr = [37.92, 27.39, 45.48, 36.69, 41.39, 57.37, 38.41, 31.47, 48.70, 44.49]     # 特征osr的10个值
# ne = [83.88, 140.78, 92.22, 112.12, 92.55, 124.22]       # 特征ne的10个值
ne = [83.88, 92.22, 112.12, 92.55, 124.22]       # 特征ne的10个值
# ne = [83.88, 140.78, 92.22, 112.12, 92.55, 164.13, 124.22]       # 特征ne的10个值
# ne = [83.88, 85.33, 140.78, 92.22, 112.12, 261.47, 92.55, 94.51, 164.13, 124.22]       # 特征ne的10个值

base_sr = [14.13, 13.75, 12.89, 15.61, 15.24]
base_osr = [20.07, 20.32, 19.58, 21.69, 20.94]
base_ne = [67.79, 71.87, 71.48, 67.49, 70.37]

def calculate_statistics_numpy(features_dict):
    """
    使用NumPy计算多个特征的统计量
    """
    results = {}

    for feature_name, values in features_dict.items():
        # 转换为NumPy数组
        arr = np.array(values)

        # 计算平均值和方差
        mean_val = np.mean(arr)
        variance_val = np.var(arr, ddof=1)  # ddof=1 计算样本方差（无偏估计）

        # 计算其他有用的统计量
        median_val = np.median(arr)      # 中位数
        std_val = np.std(arr, ddof=1)    # 标准差
        min_val = np.min(arr)            # 最小值
        max_val = np.max(arr)            # 最大值
        q1 = np.percentile(arr, 25)      # 第一四分位数
        q3 = np.percentile(arr, 75)      # 第三四分位数

        results[feature_name] = {
            '平均值': mean_val,
            '方差': variance_val,
            '标准差': std_val,
            '中位数': median_val,
            '最小值': min_val,
            '最大值': max_val,
            '范围': max_val - min_val,
        }

    return results

# 将特征组织为字典
features = {
    'sr': sr,
    'osr': osr,
    'ne': ne
}

base_features = {
    'sr': base_sr,
    'osr': base_osr,
    'ne': base_ne,
}

# 计算统计量
statistics = calculate_statistics_numpy(features)
base_statistics = calculate_statistics_numpy(base_features)


# 打印结果
print("特征统计量分析报告")
print("=" * 60)

for feature_name, stats in statistics.items():
    print(f"\n特征: {feature_name}")
    print(f"  平均值: {stats['平均值']:.4f}")
    print(f"  方差: {stats['方差']:.4f}")
    print(f"  标准差: {stats['标准差']:.4f}")
    print(f"  中位数: {stats['中位数']:.4f}")
    print(f"  范围: {stats['最小值']:.4f} ~ {stats['最大值']:.4f} (范围: {stats['范围']:.4f})")

# 打印结果
print("特征统计量分析报告2")
print("=" * 60)

for feature_name, stats in base_statistics.items():
    print(f"\n特征: {feature_name}")
    print(f"  平均值: {stats['平均值']:.4f}")
    print(f"  方差: {stats['方差']:.4f}")
    print(f"  标准差: {stats['标准差']:.4f}")
    print(f"  中位数: {stats['中位数']:.4f}")
    print(f"  范围: {stats['最小值']:.4f} ~ {stats['最大值']:.4f} (范围: {stats['范围']:.4f})")




# import numpy as np
#
#
# def calculate_without_extremes(features_dict, remove_both=True):
#     """
#     去除极端值后计算统计量
#
#     参数:
#     features_dict: 字典，键为特征名，值为数值列表
#     remove_both: True表示去除最高和最低值，False表示只去除异常值
#     """
#     results = {}
#
#     for feature_name, values in features_dict.items():
#         if len(values) < 3:
#             print(f"警告: 特征 '{feature_name}' 的数据量不足，无法去除极端值")
#             continue
#
#         # 复制数据以避免修改原始数据
#         values_array = np.array(values)
#
#         if remove_both:
#             # 去除最高和最低各一个
#             # 找到最大值和最小值的索引
#             max_idx = np.argmax(values_array)
#             min_idx = np.argmin(values_array)
#
#             # 创建掩码，排除最大值和最小值
#             mask = np.ones(len(values_array), dtype=bool)
#             mask[max_idx] = False
#             mask[min_idx] = False
#
#             filtered_values = values_array[mask]
#             removed_values = values_array[~mask]
#         else:
#             # 使用IQR方法去除异常值
#             q1 = np.percentile(values_array, 25)
#             q3 = np.percentile(values_array, 75)
#             iqr = q3 - q1
#
#             lower_bound = q1 - 1.5 * iqr
#             upper_bound = q3 + 1.5 * iqr
#
#             mask = (values_array >= lower_bound) & (values_array <= upper_bound)
#             filtered_values = values_array[mask]
#             removed_values = values_array[~mask]
#
#         # 计算统计量
#         mean_val = np.mean(filtered_values)
#         variance_val = np.var(filtered_values, ddof=1)
#         std_val = np.std(filtered_values, ddof=1)
#         median_val = np.median(filtered_values)
#         min_val = np.min(filtered_values)
#         max_val = np.max(filtered_values)
#
#         results[feature_name] = {
#             '原始数据': values_array.tolist(),
#             '原始样本量': len(values_array),
#             '过滤后数据': filtered_values.tolist(),
#             '过滤后样本量': len(filtered_values),
#             '移除的值': removed_values.tolist() if len(removed_values) > 0 else [],
#             '平均值': mean_val,
#             '方差': variance_val,
#             '标准差': std_val,
#             '中位数': median_val,
#             '最小值': min_val,
#             '最大值': max_val,
#             '范围': max_val - min_val
#         }
#
#     return results
#
#
# # 测试数据 - 请替换为您的实际数据
# # sr = [1.2, 1.5, 1.3, 1.4, 1.6, 1.7, 1.5, 1.4, 1.3, 1.6]
# # osr = [2.1, 2.3, 2.2, 2.4, 2.0, 2.5, 2.3, 2.2, 2.1, 2.4]
# # ne = [0.8, 0.9, 0.7, 0.8, 1.0, 0.9, 0.8, 0.7, 0.9, 0.8]
#
# features = {
#     'sr': sr,
#     'osr': osr,
#     'ne': ne
# }
#
# # 计算去除最高和最低值后的统计量
# stats = calculate_without_extremes(features, remove_both=True)
#
# # 打印结果
# print("去除最高和最低值后的统计量分析")
# print("=" * 60)
#
# for feature_name, result in stats.items():
#     print(f"\n特征: {feature_name}")
#     print(f"  原始数据 ({result['原始样本量']}个): {result['原始数据']}")
#     print(f"  移除的值: {result['移除的值']}")
#     print(f"  过滤后数据 ({result['过滤后样本量']}个): {result['过滤后数据']}")
#     print(f"  过滤后平均值: {result['平均值']:.4f}")
#     print(f"  过滤后方差: {result['方差']:.4f}")
#     print(f"  过滤后标准差: {result['标准差']:.4f}")
#     print(f"  过滤后中位数: {result['中位数']:.4f}")
#     print(f"  过滤后范围: {result['最小值']:.4f} ~ {result['最大值']:.4f}")
#
# # 对比原始统计量
# print("\n" + "=" * 60)
# print("对比分析: 原始 vs 去除极值后")
# print("=" * 60)
#
# for feature_name in features.keys():
#     if feature_name in stats:
#         orig_data = np.array(features[feature_name])
#         orig_mean = np.mean(orig_data)
#         orig_var = np.var(orig_data, ddof=1)
#
#         filtered_mean = stats[feature_name]['平均值']
#         filtered_var = stats[feature_name]['方差']
#
#         mean_change = ((filtered_mean - orig_mean) / orig_mean * 100) if orig_mean != 0 else 0
#         var_change = ((filtered_var - orig_var) / orig_var * 100) if orig_var != 0 else 0
#
#         print(f"\n{feature_name}:")
#         print(f"  平均值: {orig_mean:.4f} → {filtered_mean:.4f} ({mean_change:+.2f}%)")
#         print(f"  方差: {orig_var:.4f} → {filtered_var:.4f} ({var_change:+.2f}%)")