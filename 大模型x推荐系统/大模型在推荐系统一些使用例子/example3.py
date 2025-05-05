# 使用预训练语言模型（PLM）解决用户冷启动与物品冷启动问题。代码使用Hugging Face的transformers库，
# 通过加载预训练模型，提取用户兴趣特征与物品特性，并完成冷启动推荐过程。
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel

# 1. 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 2. 加载预训练模型和分词器
model_name = "bert-base-chinese"  # 使用中文BERT模型
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name).to(device)


# 3. 嵌入生成函数
def generate_embedding(text_list):
    """
    将文本列表转化为嵌入向量
    """
    embeddings = []
    for text in text_list:
        # 文本预处理
        inputs = tokenizer(text, return_tensors="pt", truncation=True,
                           padding="max_length", max_length=64).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        # 提取 [CLS] 位置的向量作为句子嵌入
        cls_embedding = outputs.last_hidden_state[:,
                        0, :].squeeze().cpu().numpy()
        embeddings.append(cls_embedding)
    return np.array(embeddings)


# 4. 用户冷启动场景
user_data = [
    "喜欢科技类书籍和AI技术",
    "对未来经济和人工智能感兴趣",
    "经常阅读与技术创新相关的文章"
]
# 生成用户特征嵌入
print("生成用户嵌入向量...")
user_embeddings = generate_embedding(user_data)
user_profile = np.mean(user_embeddings, axis=0)  # 综合用户行为生成兴趣特征
# 5. 物品冷启动场景
item_data = [
    "新书：《人工智能的未来》",
    "课程：人工智能入门讲解",
    "视频：AI如何影响未来经济发展",
    "商品：智能家居语音助手"
]
# 生成物品特征嵌入
print("生成物品嵌入向量...")
item_embeddings = generate_embedding(item_data)


# 6. 相似度计算函数
def cosine_similarity(vec1, vec2):
    """
    计算余弦相似度
    """
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))


# 计算用户与每个物品的相似度
print("\n计算用户与物品的相似度...")
similarities = [cosine_similarity(
    user_profile, item) for item in item_embeddings]
# 7. 推荐结果排序
sorted_indices = np.argsort(similarities)[::-1]  # 按相似度降序排序
recommendations = [(item_data[idx],
                    similarities[idx]) for idx in sorted_indices]
# 8. 显示推荐结果
print("\n推荐结果：")
for i, (desc, score) in enumerate(recommendations):
    print(f"推荐物品 {i + 1}: {desc} (相似度: {score:.4f})")
# 9. 多用户冷启动模拟（扩展）
new_users = [
    ["对人工智能研究感兴趣", "喜欢学习科技创新知识"],
    ["阅读过多篇关于机器学习的文章", "经常关注技术热点新闻"],
]
# 为每位新用户生成推荐
print("\n多用户推荐结果：")
for user_id, user_behavior in enumerate(new_users):
    user_embeddings = generate_embedding(user_behavior)
    user_profile = np.mean(user_embeddings, axis=0)
    similarities = [cosine_similarity(
        user_profile, item) for item in item_embeddings]
    sorted_indices = np.argsort(similarities)[::-1]
    recommendations = [(item_data[idx],
                        similarities[idx]) for idx in sorted_indices]
    print(f"\n用户 {user_id + 1} 的推荐结果：")
    for i, (desc, score) in enumerate(recommendations):
        print(f"推荐物品 {i + 1}: {desc} (相似度: {score:.4f})")
