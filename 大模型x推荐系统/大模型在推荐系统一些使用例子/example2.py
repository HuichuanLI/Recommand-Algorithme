# 通过用户行为序列建模生成用户兴趣表示，并与物品特性进行匹配推荐。代码基于PyTorch实现。
import torch
import numpy as np
from transformers import BertTokenizer, BertModel, BertConfig

# 1. 定义设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 2. 加载预训练的BERT模型和分词器
model_name = "bert-base-uncased"  # 可以替换为其他Transformer模型
tokenizer = BertTokenizer.from_pretrained(model_name)
config = BertConfig.from_pretrained(model_name)
model = BertModel.from_pretrained(model_name, config=config).to(device)
# 3. 用户行为序列数据
user_behavior = [
    "浏览了轻便跑步鞋",
    "查看了跑步技巧文章",
    "搜索了健身服装推荐"
]
# 4. 商品描述数据
item_descriptions = [
    "轻便跑鞋，适合长跑使用",
    "透气健身服，适合运动训练",
    "高性能智能手表，监测健康数据"
]


# 5. 文本嵌入生成函数
def generate_embedding(text_list):
    embeddings = []
    for text in text_list:
        inputs = tokenizer(text, return_tensors="pt", truncation=True,
                           padding="max_length", max_length=64).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        # 使用 [CLS] 位置的向量作为句子嵌入
        cls_embedding = \
            outputs.last_hidden_state[:, 0, :].squeeze().cpu().numpy()
        embeddings.append(cls_embedding)
    return np.array(embeddings)


# 6. 生成用户行为嵌入
print("生成用户行为嵌入...")
user_embeddings = generate_embedding(user_behavior)
user_profile = np.mean(user_embeddings, axis=0)  # 平均用户行为向量表示用户兴趣
# 7. 生成商品嵌入
print("生成商品描述嵌入...")
item_embeddings = generate_embedding(item_descriptions)


# 8. 计算用户兴趣与商品的相似度
def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))


print("\n计算相似度并推荐商品...")
similarities = [cosine_similarity(user_profile,
                                  item) for item in item_embeddings]
# 9. 推荐结果排序
sorted_indices = np.argsort(similarities)[::-1]  # 按相似度降序排序
recommendations = [(item_descriptions[idx],
                    similarities[idx]) for idx in sorted_indices]
# 10. 输出推荐结果
for i, (desc, score) in enumerate(recommendations):
    print(f"推荐商品 {i + 1}: {desc} (相似度: {score:.4f})")
