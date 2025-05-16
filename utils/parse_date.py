import re
import datetime
import os

def extract_comic_series_and_date(filename):
    """
    从文件名中提取漫画集名称和日期
    
    Args:
        filename: 文件名，如 "Fred Basset_2022-09-09.jpg" 或 "Fred Basset en Español_2025-03-25.jpg"
        
    Returns:
        tuple: (漫画集名称, 日期字符串, 日期对象)，如果无法提取日期则返回(漫画集名称, None, None)
    """
    # 尝试匹配常见的日期格式 (YYYY-MM-DD)
    # date_pattern = r'_(\d{4}-\d{2}-\d{2})\.'
    date_pattern = r'_(\d{4}-\d{2}-\d{2})(?:\..*)?$'
    match = re.search(date_pattern, filename)
    
    if match:
        # 提取日期字符串
        date_str = match.group(1)
        
        # 提取漫画集名称（日期前的部分）
        series_name = filename.split('_')[0]
        try:
            # 转换为日期对象
            date_obj = datetime.datetime.strptime(date_str, '%Y-%m-%d').date()
            return series_name, date_str, date_obj
        except ValueError:
            # 日期格式错误
            return series_name, date_str, None
    
    # 如果没有找到标准日期格式，尝试其他格式
    # 例如 "Series Name_YYYYMMDD.jpg"
    # date_pattern2 = r'_(\d{8})\.'
    date_pattern2 = r'_(\d{8})(?:\..*)?$'
    match = re.search(date_pattern2, filename)
    
    if match:
        date_str = match.group(1)
        try:
            # 转换为标准格式
            date_obj = datetime.datetime.strptime(date_str, '%Y%m%d').date()
            formatted_date = date_obj.strftime('%Y-%m-%d')
            series_name = filename.split('_')[0]
            return series_name, formatted_date, date_obj
        except ValueError:
            # 日期格式错误
            return filename.split('_')[0], date_str, None
    
    # 没有找到日期，可能是其他格式
    parts = os.path.splitext(filename)[0].split('_')
    if len(parts) > 1:
        return parts[0], parts[1], None
    
    # 无法提取，返回文件名作为系列名
    return filename, None, None