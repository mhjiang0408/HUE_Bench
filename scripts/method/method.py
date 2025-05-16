import os
import sys
sys.path.append(os.getcwd())
from utils.llm_call import CallLLM
import base64
from utils.base64_encoder import encode_image_to_base64

class reasoning_thinker():
    def __init__(self,model_name="qwq-32b",api_base:str='https://api.openai.com/v1', api_key:str='sk-xxx'):
        self.llm = CallLLM(model=model_name,api_base=api_base,api_key=api_key)

    def reasoning_thinking(self, description:str):
        system_prompt = "Following the above political comic description, you need to analyze which of the option images presents the same humor expression as the reference image (i.e., originates from the same cartoonist). Answer AS SIMPLE AS POSSIBLE. Make sure the probabilities add up to 1.\n # Response Format\n ```json\n { \"A\": probability of choosing the option A, \"B\": probability of choosing the option B, \"C\": probability of choosing the option C, \"D\": probability of choosing the option D }\n```"

        user_prompt = f"The options are the images. The image descriptions are: {description}. The answer in image descriptions may be wrong. My question is: Which of the option images presents the same humor expression as the reference image? First, analyze the social background or event behind each image, using this context to interpret the comic. When determining which image shares the same humor expression as the reference image, please focus on how the cartoon conveys its viewpoint, the specific techniques used to construct humor, and how visual elements are employed to express that humor. Please predict the probability that you would choose each option."
        message = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        response, total_prompt_tokens, total_completion_tokens, reasoning_content = self.llm.post_reasoning_request(message)
        return response, total_prompt_tokens, total_completion_tokens, reasoning_content

class description_extractor():
    def __init__(self,model_name="qwen2.5-vl-7b-instruct", low_detail:bool=False, api_base:str='https://api.openai.com/v1', api_key:str='sk-xxx'):
        self.llm = CallLLM(model=model_name,api_base=api_base,api_key=api_key)
        self.low_detail = low_detail
        self.option_ids = ['A', 'B', 'C', 'D']
    def extract_description(self, reference_image:str, options_image_paths:dict):

        # 因为使用的是处理后的图像，所以需要将image_path解析后更换到目录下

        system_prompt = "I will provide you with a political comic. Please give a detailed explanation of the visual elements in the image and analyze its humor expression. You must faithfully describe the information presented in the image without making any speculation or subjective judgments. Your description should cover the following aspects: - First, provide a detailed account of all the visual elements in the image and their illustrative characteristics.\n- Then, describe how the different elements are combined, giving a thorough depiction of the scene portrayed in the comic and the potential sociopolitical context it references.\n- Finally, explain the comic’s humor punchline and analyze how the humor is conveyed. Your answer should begin with 'The reference image shows', 'The image A shows', 'The image B shows', 'The image C shows', and 'The image D shows'. Answer the more the better."
        user_prompt = "I'm blind. Here is the image. Please think step-by-step and describe the political comic in detail and present your answer in the form of a Pseudo-CoT. After describing each image, please analyze which of the option images presents the same humor expression style as the reference image (i.e., originates from the same cartoonist), based on similarities in viewpoint expression, humor construction techniques, and visual humor illustration style. Think step by step but DO NOT give your final answer. The image above is the reference image. The following images provided to you are, in order, A, B, C, and D."
        reference_image_base64 = encode_image_to_base64(reference_image)
        if not reference_image_base64:
            print(f"错误：无法编码参考图像 {reference_image}")
            return None
        option_images_base64 = {}
        for opt_id, opt_path in options_image_paths.items():
            encoded = encode_image_to_base64(opt_path)
            if not encoded:
                print(f"错误：无法编码选项图像 {opt_path} (选项 {opt_id})")
                return None # 如果任何一个选项图像失败，则无法构建消息
            option_images_base64[opt_id] = encoded
        user_content = [
            {
                "type": "image_url",
                "image_url": {"url": reference_image_base64, "detail": "low"}
            },
            {
                "type": "text",
                "text": user_prompt
            }
        ]
        for opt_id in self.option_ids:
            if opt_id in option_images_base64:
                user_content.append({
                    "type": "image_url",
                    "image_url": {"url": option_images_base64[opt_id], "detail": "low"}
                })
            else:
                print(f"警告：缺少选项 {opt_id} 的图像数据。")
                return None  # 必须有所有选项才能继续
        
        message = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content}
        ]
        response, total_prompt_tokens, total_completion_tokens = self.llm.post_request(message)
        return response, total_prompt_tokens, total_completion_tokens

class CoVR():
    def __init__(self,mllm_model="qwen2.5-vl-7b-instruct", reasoning_model="deepseek-r1-250120", low_detail:bool=False,mllm_api_base:str='https://api.openai.com/v1',mllm_api_key:str='sk-xxx',reasoning_api_base:str='https://api.openai.com/v1',reasoning_api_key:str='sk-xxx'):
        self.description_agent = description_extractor(mllm_model, low_detail,mllm_api_base,mllm_api_key)
        self.reasoning_agent = reasoning_thinker(reasoning_model,reasoning_api_base,reasoning_api_key)
    def post_request(self,options_image_paths:dict,reference_image:str):
        # options = '\n'.join([f'{opt_id}: {opt_content}' for opt_id,opt_content in options])

        description, total_prompt_tokens, total_completion_tokens = self.description_agent.extract_description(reference_image, options_image_paths)
        response, total_reasoning_prompt_tokens, total_reasoning_completion_tokens, reasoning_content = self.reasoning_agent.reasoning_thinking(description)
        # 将description放到option中
        return response, f'{total_prompt_tokens};{total_completion_tokens}',f'{total_reasoning_prompt_tokens};{total_reasoning_completion_tokens}', total_prompt_tokens+total_completion_tokens+total_reasoning_prompt_tokens+total_reasoning_completion_tokens,f'{{"description":"{description}","options":"{options_image_paths}","reasoning_content":"{reasoning_content}"}}'
    
    def post_existing_request(self,options_image_paths:dict,description:str):
        response, total_reasoning_prompt_tokens, total_reasoning_completion_tokens, reasoning_content = self.reasoning_agent.reasoning_thinking(description)
        return response, f'{total_reasoning_prompt_tokens};{total_reasoning_completion_tokens}',f'{total_reasoning_prompt_tokens};{total_reasoning_completion_tokens}', total_reasoning_prompt_tokens+total_reasoning_completion_tokens,f'{{"description":"{description}","options":"{options_image_paths}","reasoning_content":"{reasoning_content}"}}'
