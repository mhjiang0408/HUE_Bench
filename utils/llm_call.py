import time
from functools import wraps
import openai
from curl_cffi import requests

def retry_on_failure(max_retries=3, delay=1):
    """
    重试装饰器
    
    Args:
        max_retries: 最大重试次数
        delay: 重试间隔（秒）
    """
    def decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            retries = 0
            last_error = None
            
            while retries < max_retries:
                try:
                    return func(self, *args, **kwargs)
                except Exception as e:
                    last_error = e
                    retries += 1
                    if retries < max_retries:
                        print(f"模型 {self.model} 请求失败 (尝试 {retries}/{max_retries}): {str(e)}")
                        print(f"{delay}秒后重试...")
                        time.sleep(delay)
            
            print(f"模型 {self.model} 重试{max_retries}次后仍然失败: {str(last_error)}")
            raise last_error
            
        return wrapper
    return decorator

class CallLLM:
    def __init__(self, model:str = "Qwen/Qwen2.5-7B-Instruct", 
                 api_base:str = "http://cn.api.beer/v1", 
                 api_key:str = "sk-Z0MdU0NAXCmiwYF_GjMe5rCO_2iFNU_FuPnS7jdcge54rdYa2yRnF6S9ngk"):
        self.model = model
        self.api_base = api_base
        self.api_key = api_key
        
    @retry_on_failure(max_retries=20, delay=4)
    def post_request(self, messages:list) -> tuple[str, int, int]:
        """
        发送请求并获取答案，带有重试机制
        """
        client = openai.OpenAI(base_url=self.api_base,api_key=self.api_key)
        if 'gemini' in self.model:
            response = client.chat.completions.create(
                model=self.model,
                reasoning_effort="none",
                messages=messages
            )
        else:
            response = client.chat.completions.create(
                model=self.model,
                messages=messages,
            )
        return response.choices[0].message.content, response.usage.prompt_tokens, response.usage.completion_tokens
    
    @retry_on_failure(max_retries=20, delay=4)
    def post_reasoning_request(self, messages:list) -> tuple[str, int,int, str]:
        """
        发送请求并获取答案，带有重试机制
        """
        client = openai.OpenAI(base_url=self.api_base,api_key=self.api_key)
        if self.model == "o3-mini":
            
            response = client.chat.completions.create(
                model="o3-mini",
                reasoning_effort="high",
                messages=messages
            )
            return response.choices[0].message.content, response.usage.prompt_tokens, response.usage.completion_tokens, response.choices[0].message.content
        else:
            reasoning_content = ""
            answer_content = ""

            # 创建聊天完成请求
            completion = client.chat.completions.create(
                model=self.model,  # 使用实例的model属性
                messages=messages,
                stream=True,
                # 启用usage统计
                stream_options={
                    "include_usage": True
                }
            )
            usage_info = None
            for chunk in completion:
                # 如果chunk.choices为空，则记录usage
                if not chunk.choices:
                    usage_info = chunk.usage
                else:
                    delta = chunk.choices[0].delta
                    # 收集思考过程
                    if hasattr(delta, 'reasoning_content') and delta.reasoning_content is not None:
                        reasoning_content += delta.reasoning_content
                    # 收集回答内容
                    elif hasattr(delta, 'content') and delta.content is not None:
                        answer_content += delta.content

            # 打印最终的完整回答
            # if answer_content:
            #     # print("\n" + "=" * 20 + "完整回复" + "=" * 20 + "\n")
            #     print(answer_content)
            # print("\n" + "=" * 20 + "推理过程" + "=" * 20 + "\n")
            # print(reasoning_content)
            # 打印token使用统计
            # if usage_info:
                # print("\n" + "=" * 20 + "Token统计" + "=" * 20)
                # print(f"提示tokens: {usage_info.prompt_tokens}")
                # print(f"补全tokens: {usage_info.completion_tokens}")
                # print(f"总计tokens: {usage_info.total_tokens}")
            return answer_content, usage_info.prompt_tokens, usage_info.completion_tokens, reasoning_content
    