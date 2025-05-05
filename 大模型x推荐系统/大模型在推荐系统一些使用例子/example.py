# 使用预训练语言模型（PLM）进行用户与物品的联合建模，代码基于Hugging Face的transformers库，
# 完成从用户兴趣提取、物品特性建模到向量匹配计算的全过程。

from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np

# 1. 加载预训练模型和分词器
model_name = "distilbert-base-uncased"  # 可替换为其他PLM模型
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


# 2. 定义文本嵌入生成函数
def get_text_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding="max_length", max_length=64).to(device)
    outputs = model(**inputs)
    # 提取最后一层的[CLS]向量作为嵌入
    embedding = outputs.last_hidden_state[:, 0, :].squeeze().detach().cpu().numpy()
    return embedding


# 3. 定义用户兴趣和物品描述
user_behavior = [
    "适合长跑的跑鞋",
    "跑步装备推荐",
    "如何挑选缓震跑鞋"
]
item_descriptions = [
    "轻便跑鞋，适合长距离跑步",
    "户外徒步鞋，防水防滑设计",
    "缓震跑鞋，适合长时间运动"
]
# 4. 生成用户兴趣嵌入
user_embeddings = [get_text_embedding(text) for text in user_behavior]
# 计算用户的整体兴趣向量（取平均值）
user_profile = np.mean(user_embeddings, axis=0)
# 5. 生成物品特性嵌入
item_embeddings = [get_text_embedding(text) for text in item_descriptions]


# 6. 计算用户与物品的相似度
def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))


similarities = [cosine_similarity(user_profile, item) for item in item_embeddings]
# 7. 打印推荐结果
print("用户兴趣嵌入向量:")
print(user_profile)
print("\n物品描述及其相似度:")
for idx, desc in enumerate(item_descriptions):
    print(f"物品描述: {desc}")
    print(f"相似度: {similarities[idx]:.4f}")
