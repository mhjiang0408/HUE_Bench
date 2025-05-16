import pandas as pd
import glob
import os

# 指定文件夹路径
folder_path = './Data/statistics/journals'  # 替换为你的文件夹路径

# 获取文件夹中所有的CSV文件
csv_files = glob.glob(os.path.join(folder_path, '*.csv'))

# 读取并合并所有CSV文件
df_list = []
for file in csv_files:
    # 读取CSV文件
    df = pd.read_csv(file)
    df_list.append(df)

# 合并所有数据框
merged_df = pd.concat(df_list, ignore_index=True)

# 保存合并后的文件
output_path = os.path.join(folder_path, 'merged_output.csv')
merged_df.to_csv(output_path, index=False)