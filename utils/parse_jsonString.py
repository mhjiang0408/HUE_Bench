import json
import re

def parse_probabilities(text: str, num_options: int = 4) -> dict:
    """
    从字符串中解析选项概率，支持字符串形式的概率值
    
    Args:
        text: JSON格式的字符串，包含选项概率
        num_options: 选项数量，如果为None则自动检测
        
    Returns:
        dict: 包含选项概率的字典，如果解析失败返回None
    """
    # 清理字符串
    text = text.replace("'", '"')
    text = text.replace('```json', '').replace('```', '')
    text = text.strip()
    valid_options = [chr(65 + i) for i in range(num_options)]
    
    try:
        # 尝试直接解析JSON
        probabilities = json.loads(text)
        
        if isinstance(probabilities, str):
            probabilities = json.loads(probabilities)
        if not isinstance(probabilities, dict):
            raise json.JSONDecodeError("probabilities is not a dict", text, 0)
        
        # 验证数据格式
        if not all(key in valid_options for key in probabilities.keys()):
            print(f"选项格式错误，期望选项: {valid_options}，实际选项: {list(probabilities.keys())}")
            return None
        
        # 转换所有值为浮点数
        converted_probs = {}
        for key, value in probabilities.items():
            try:
                # 处理字符串形式的数值
                if isinstance(value, str):
                    # 移除可能的引号和空格
                    value = value.strip('"\'').strip()
                    # 转换为浮点数
                    converted_probs[key] = float(value)
                else:
                    converted_probs[key] = float(value)
            except (ValueError, TypeError):
                print(f"无法将值转换为浮点数: {key}={value}")
                return None
            
        # 如果指定了选项数量，确保所有选项都存在
        if num_options:
            missing_options = set(valid_options) - set(converted_probs.keys())
            if missing_options:
                print(f"选项数量不匹配，补全缺失的选项: {missing_options}")
                for option in missing_options:
                    converted_probs[option] = 0.0  # 将缺失选项的概率设为0
        
        return converted_probs
        
    except json.JSONDecodeError:
        # 如果JSON解析失败，尝试使用正则表达式提取
        pattern = r'"([A-Z])"\s*:\s*"?(\d+(?:\.\d+)?)"?'
        matches = re.findall(pattern, text)
        
        if not matches:
            print("无法解析答案格式")
            return None
            
        # 转换所有值为浮点数
        probabilities = {}
        for option, value in matches:
            try:
                probabilities[option] = float(value)
            except ValueError:
                print(f"无法将值转换为浮点数: {option}={value}")
                return None
        
        # 验证提取的数据
        if num_options:
            missing_options = set(valid_options) - set(probabilities.keys())
            if missing_options:
                print(f"选项数量不匹配，补全缺失的选项: {missing_options}")
                for option in missing_options:
                    probabilities[option] = 0.0  # 将缺失选项的概率设为0
        
        return probabilities
    
def parse_json_string(text: str, expected_keys: list = None, validate_types: dict = None) -> dict:
    """
    从字符串中解析任意JSON格式数据
    
    Args:
        text: 包含JSON数据的字符串
        expected_keys: 可选，期望JSON中包含的键列表
        validate_types: 可选，键值对字典，指定特定键的值应该是什么类型
        
    Returns:
        dict: 解析后的JSON字典，如果解析失败返回None
    """
    # 清理字符串，移除可能的代码块标记
    if '```' in text:
        # 处理可能的代码块
        text = re.sub(r'```(?:json|python|javascript)?', '', text)
        text = text.replace('```', '')
    
    text = text.strip()
    
    try:
        # 尝试直接解析JSON
        parsed_data = json.loads(text)
        
        # 如果不是字典类型，包装成字典
        if not isinstance(parsed_data, dict):
            print(f"警告：解析结果不是字典类型，而是 {type(parsed_data).__name__}")
            return {"value": parsed_data}
        
        # 验证是否包含所有期望的键
        if expected_keys:
            missing_keys = [key for key in expected_keys if key not in parsed_data]
            if missing_keys:
                print(f"缺少期望的键: {missing_keys}")
                return None
        
        # 验证值的类型
        if validate_types:
            for key, expected_type in validate_types.items():
                if key in parsed_data:
                    if not isinstance(parsed_data[key], expected_type):
                        actual_type = type(parsed_data[key]).__name__
                        expected_type_name = expected_type.__name__
                        print(f"键 '{key}' 的值类型错误，期望 {expected_type_name}，实际 {actual_type}")
                        return None
        
        return parsed_data
        
    except json.JSONDecodeError as e:
        print(f"JSON解析错误: {str(e)}")
        
        # 尝试修复常见的JSON格式问题
        try:
            # 尝试修复单引号问题
            fixed_text = text.replace("'", '"')
            return json.loads(fixed_text)
        except:
            pass
            
        try:
            # 为没有引号的键添加引号
            fixed_text = re.sub(r'(\s*)(\w+)(\s*):', r'\1"\2"\3:', text)
            return json.loads(fixed_text)
        except:
            pass
        
        # 如果是选项概率格式，尝试使用正则表达式提取
        option_pattern = r'"([A-Za-z0-9_]+)"\s*:\s*(\d+(?:\.\d+)?)'
        matches = re.findall(option_pattern, text)
        
        if matches:
            parsed_data = {option: float(value) for option, value in matches}
            return parsed_data
            
        return None
    
    except Exception as e:
        print(f"解析过程中发生错误: {str(e)}")
        return None