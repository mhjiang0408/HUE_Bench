import time
import openai
import requests
def test_model_api(model_name:str, api_base, api_key):
    if "CoVR" in model_name or "clip" in model_name or "few_shot" in model_name:
        print(f"✅ API测试成功: {model_name}")
        return True
    if "ocr" or "majority" or "few_shot" in model_name:
        model_name = model_name.replace("_ocr", "")
        model_name = model_name.replace("_majority", "")
        model_name = model_name.replace("_few_shot_1", "").replace("_few_shot_2", "").replace("_few_shot_5", "")
    client = openai.OpenAI(
        api_key=api_key,
        base_url=api_base
    )
    
    retries = 0
    while True:  # 无限循环，直到API可用
        try:
            # 发送一个简单的测试请求
            response = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": "Hello"}],
                max_tokens=5
            )
            print(f"✅ API测试成功: {model_name}")
            return True
        except (openai.APIError, openai.APIConnectionError, 
                openai.RateLimitError, requests.exceptions.RequestException) as e:
            retries += 1
            wait_time = 5  # 逐渐增加等待时间，但最多等待60秒
            print(f"❌ API测试失败 (尝试 #{retries}): {model_name} - {str(e)}")
            print(f"⏳ 等待 {wait_time} 秒后重试...")
            time.sleep(wait_time)
        except Exception as e:
            print(f"❌ 未知错误: {model_name} - {str(e)}")
            print("⏳ 等待 5 秒后重试...")
            time.sleep(5)
            # 继续尝试，不返回 False