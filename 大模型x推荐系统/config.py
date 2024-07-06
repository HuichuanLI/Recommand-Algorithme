58xueke.com
import os
from dotenv import load_dotenv, find_dotenv

# 加载环境变量
load_dotenv(find_dotenv())

# 获取 OpenAI API 密钥
OPENAI_API_KEY = os.environ['OPENAI_API_KEY']

# 官方文档 - Models：https://platform.openai.com/docs/models
MODELS = [
    "gpt-3.5-turbo-1106", # 最新的 GPT-3.5 Turbo 模型，具有改进的指令遵循、JSON 模式、可重现输出、并行函数调用等。最多返回 4,096 个输出标记。
    "gpt-3.5-turbo",  # 当前指向 gpt-3.5-turbo-0613 。自 2023 年 12 月 11  日开始指向gpt-3.5-turbo-1106。
    "gpt-3.5-turbo-16k",  # 当前指向 gpt-3.5-turbo-0613 。将指向gpt-3.5-turbo-1106 2023 年 12 月 11 日开始。
    # "gpt-3.5-turbo-instruct",  # 与 text-davinci-003功能类似，但兼容遗留的 Completions 端点，而不是 Chat Completions。
    "gpt-4-1106-preview", # 最新的 GPT-4 模型，具有改进的指令跟踪、JSON 模式、可重现输出、并行函数调用等。最多返回 4,096 个输出标记。此预览模型尚不适合生产流量。
    # "gpt-4-vision-preview", # 除了所有其他 GPT-4 Turbo 功能外，还能够理解图像。最多返回 4,096 个输出标记。这是一个预览模型版本，尚不适合生产流量。
    "gpt-4",  # 当前指向 gpt-4-0613。8192 tokens
    # "gpt-4-32k",  # 当前指向 gpt-4-32k-0613。32768 tokens
    # "gpt-4-0613", # 从 2023 年 6 月 13 日开始的 gpt-4 快照，改进了函数调用支持。
    # "gpt-4-32k-0613", # 从 2023 年 6 月 13 日开始的 gpt-4-32k 快照，改进了函数调用支持。
]
DEFAULT_MODEL = MODELS[-2]
MODEL_TO_MAX_TOKENS = {
    "gpt-3.5-turbo-1106": 4096,
    "gpt-3.5-turbo": 4096,
    "gpt-3.5-turbo-16k": 16385,
    "gpt-4-1106-preview": 4096,
    "gpt-4": 8192,
}

QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
QDRANT_EMBEDDING_DIMS = 768