import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity

# 1. 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 2. 加载预训练模型
model_name = "bert-base-chinese"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name).to(device)
from sklearn.metrics import accuracy_score
import torch
import torch.nn as nn
import torch.optim as optim
from transformers import AutoModel, AutoTokenizer
from torch.utils.data import Dataset, DataLoader


# 1. 加载MIND数据集
def load_mind_dataset(news_path, behaviors_path):
    """
    加载新闻数据和用户行为数据
    """
    # 加载新闻数据
    news_df = pd.read_csv(news_path, sep='\t', header=None,
                          names=['NewsID', 'Category', 'SubCategory', 'Title',
                                 'Abstract', 'URL', 'TitleEntities', 'AbstractEntities'])
    print("新闻数据示例：")
    print(news_df.head())
    # 加载用户行为数据
    behaviors_df = pd.read_csv(behaviors_path, sep='\t', header=None,
                               names=['ImpressionID', 'UserID', 'Time', 'History', 'Impressions'])
    print("\n用户行为数据示例：")
    print(behaviors_df.head())

    return news_df, behaviors_df


# 2. 数据清洗与缺失值处理
def clean_data(news_df, behaviors_df):
    """
    清洗数据，处理缺失值
    """
    print("\n清洗前新闻数据缺失值：")
    print(news_df.isnull().sum())
    # 填充缺失值
    news_df.fillna("", inplace=True)
    behaviors_df.fillna("", inplace=True)
    print("\n清洗后新闻数据缺失值：")
    print(news_df.isnull().sum())
    return news_df, behaviors_df


# 3. 特征提取与处理
def extract_features(news_df, behaviors_df):
    """
    提取标题TF-IDF特征和用户历史行为特征
    """
    # 提取新闻标题TF-IDF特征
    vectorizer = TfidfVectorizer(max_features=1000)
    news_df['TitleTFIDF'] = list(vectorizer.fit_transform(
        news_df['Title']).toarray())
    print("\n标题TF-IDF特征示例：")
    print(news_df[['Title', 'TitleTFIDF']].head())
    # 处理用户历史行为
    behaviors_df['HistoryList'] = behaviors_df['History'].apply(
        lambda x: x.split() if x != "" else [])
    print("\n用户历史行为示例：")
    print(behaviors_df[['UserID', 'HistoryList']].head())
    return news_df, behaviors_df


# 4. 标签编码与数据分割
def encode_and_split(news_df, behaviors_df):
    """
    对类别数据进行编码，并划分训练集与测试集
    """
    # 类别编码
    le = LabelEncoder()
    news_df['CategoryEncoded'] = le.fit_transform(news_df['Category'])
    print("\n新闻类别编码示例：")
    print(news_df[['Category', 'CategoryEncoded']].head())
    # 用户历史分割为训练集和测试集
    train_behaviors, test_behaviors = train_test_split(
        behaviors_df, test_size=0.2, random_state=42)
    print("\n训练集与测试集示例：")
    print("训练集：", train_behaviors.shape)
    print("测试集：", test_behaviors.shape)
    return news_df, train_behaviors, test_behaviors


# 3. 嵌入生成函数
def generate_embedding(text_list):
    """
    输入一组文本，返回其嵌入向量
    """
    embeddings = []
    for text in text_list:
        inputs = tokenizer(text, return_tensors="pt", truncation=True,
                           padding="max_length", max_length=64).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        # 使用[CLS]向量作为文本的嵌入
        cls_embedding = outputs.last_hidden_state[:,
                        0, :].squeeze().cpu().numpy()
        embeddings.append(cls_embedding)
    return np.array(embeddings)


# 4. 模拟用户和物品数据
def create_sample_data():
    """
    创建用户兴趣描述和物品描述样本
    """
    user_descriptions = [
        "喜欢科幻类书籍，关注人工智能和未来科技",
        "热爱历史，偏好文学与文化类书籍",
        "喜欢金融类书籍，研究股票与市场趋势"
    ]
    book_descriptions = [
        "《未来的人工智能》：探讨AI技术的潜力",
        "《股票市场动态》：全面解读股市发展",
        "《历史的长河》：聚焦古代文明与文化",
        "《深度学习实践》：神经网络与算法案例",
        "《文学与生活》：解读经典文学与现代社会"
    ]
    return user_descriptions, book_descriptions


# 5. 用户与物品特征嵌入生成
def generate_user_item_embeddings(user_descriptions, book_descriptions):
    """
    生成用户和物品的嵌入向量
    """
    print("生成用户嵌入向量...")
    user_embeddings = generate_embedding(user_descriptions)
    print("生成物品嵌入向量...")
    book_embeddings = generate_embedding(book_descriptions)
    return user_embeddings, book_embeddings


# 6. 相似度计算
def calculate_similarity(user_embeddings, book_embeddings,
                         user_descriptions, book_descriptions):
    """
    计算用户与物品嵌入的相似度
    """
    for i, user_embedding in enumerate(user_embeddings):
        similarities = cosine_similarity([user_embedding],
                                         book_embeddings)[0]
        print(f"\n用户兴趣描述: {user_descriptions[i]}")
        print("推荐结果：")
        sorted_indices = np.argsort(similarities)[::-1]
        for rank, idx in enumerate(sorted_indices):
            print(f"推荐书籍 {rank + 1}: {book_descriptions[idx]} 相似度: {similarities[idx]: .4f})")


class RecommendationDataset(Dataset):
    """
    自定义数据集，包含用户兴趣和物品描述
    """

    def __init__(self, user_texts, item_texts, labels,
                 tokenizer, max_length=64):
        self.user_texts = user_texts
        self.item_texts = item_texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        user_input = self.tokenizer(
            self.user_texts[idx],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        item_input = self.tokenizer(
            self.item_texts[idx],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        label = torch.tensor(self.labels[idx], dtype=torch.float)
        return user_input, item_input, label


# 3. 预训练推荐模型定义
class PretrainedRecommendationModel(nn.Module):
    """
    使用BERT作为用户和物品特征提取器
    """

    def __init__(self, model_name):
        super(PretrainedRecommendationModel, self).__init__()
        self.user_model = AutoModel.from_pretrained(model_name)
        self.item_model = AutoModel.from_pretrained(model_name)
        self.classifier = nn.Sequential(
            nn.Linear(768 * 2, 512),
            nn.ReLU(),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )

    def forward(self, user_input, item_input):
        user_embedding = self.user_model(**user_input).pooler_output
        item_embedding = self.item_model(**item_input).pooler_output
        combined = torch.cat((user_embedding, item_embedding), dim=1)
        output = self.classifier(combined)
        return output


# 4. 模拟训练数据
def create_sample_data():
    user_texts = [
        "喜欢科幻类书籍，关注人工智能和未来科技",
        "热爱历史，偏好文学与文化类书籍",
        "喜欢金融类书籍，研究股票与市场趋势"
    ]
    item_texts = [
        "《未来的人工智能》：探讨AI技术的潜力",
        "《股票市场动态》：全面解读股市发展",
        "《历史的长河》：聚焦古代文明与文化"
    ]
    labels = [1, 1, 0]  # 假设用户对物品1和2有兴趣，对物品3无兴趣
    return user_texts, item_texts, labels


# 5. 训练函数
def train_model(model, data_loader, optimizer, criterion, epochs=3):
    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for user_input, item_input, labels in data_loader:
            user_input = {key: val.squeeze(0).to(device) for key,
            val in user_input.items()}
            item_input = {key: val.squeeze(0).to(device) for key,
            val in item_input.items()}
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(user_input, item_input)
            loss = criterion(outputs.squeeze(), labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"第 {epoch + 1} 轮训练完成，平均损失：{total_loss / len(data_loader): .4f}")
        # 6. 测试函数


def evaluate_model(model, data_loader):
    model.eval()
    all_labels = []
    all_predictions = []
    with torch.no_grad():
        for user_input, item_input, labels in data_loader:
            user_input = {key: val.squeeze(0).to(device) for key,
            val in user_input.items()}
            item_input = {key: val.squeeze(0).to(device) for key,
            val in item_input.items()}
            labels = labels.numpy()
            outputs = model(user_input, item_input)
            predictions = (outputs.squeeze().cpu().numpy() > 0.5).astype(int)
            all_labels.extend(labels)
            all_predictions.extend(predictions)
    accuracy = accuracy_score(all_labels, all_predictions)
    print(f"测试集准确率：{accuracy:.4f}")


class RecommendationDataset(Dataset):
    """
    定义推荐系统推理数据集
    """

    def __init__(self, user_texts, item_texts, labels,
                 tokenizer, max_length=64):
        self.user_texts = user_texts
        self.item_texts = item_texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        user_input = self.tokenizer(
            self.user_texts[idx],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        item_input = self.tokenizer(
            self.item_texts[idx],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        label = torch.tensor(self.labels[idx], dtype=torch.float)
        return user_input, item_input, label


# 3. 加载预训练模型
class PretrainedRecommendationModel(torch.nn.Module):
    """
    使用BERT提取用户和物品特征
    """

    def __init__(self, model_name):
        super(PretrainedRecommendationModel, self).__init__()
        self.user_model = AutoModel.from_pretrained(model_name)
        self.item_model = AutoModel.from_pretrained(model_name)
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(768 * 2, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 1),
            torch.nn.Sigmoid()
        )

    def forward(self, user_input, item_input):
        user_embedding = self.user_model(**user_input).pooler_output
        item_embedding = self.item_model(**item_input).pooler_output
        combined = torch.cat((user_embedding, item_embedding), dim=1)
        output = self.classifier(combined)
        return output


# 4. 模拟推理数据
def create_sample_data():
    user_texts = [
        "喜欢科幻类书籍，关注人工智能和未来科技",
        "热爱历史，偏好文学与文化类书籍",
        "喜欢金融类书籍，研究股票与市场趋势"
    ]
    item_texts = [
        "《未来的人工智能》：探讨AI技术的潜力",
        "《股票市场动态》：全面解读股市发展",
        "《历史的长河》：聚焦古代文明与文化"
    ]
    labels = [1, 0, 1]  # 假设标签，1表示感兴趣，0表示不感兴趣
    return user_texts, item_texts, labels


# 5. 推理函数
def inference(model, data_loader):
    """
    推理用户与物品的匹配度
    """
    model.eval()
    predictions, true_labels = [], []
    with torch.no_grad():
        for user_input, item_input, labels in data_loader:
            user_input = {key: val.squeeze(0).to(device) for key,
            val in user_input.items()}
            item_input = {key: val.squeeze(0).to(device) for key,
            val in item_input.items()}
            labels = labels.numpy()
            outputs = model(user_input, item_input)
            preds = (outputs.squeeze().cpu().numpy() > 0.5).astype(int)
            predictions.extend(preds)
            true_labels.extend(labels)
    return predictions, true_labels


# 6. 评估函数
def evaluate(predictions, true_labels):
    """
    评估推荐系统性能
    """
    accuracy = accuracy_score(true_labels, predictions)
    report = classification_report(true_labels, predictions,
                                   target_names=["不感兴趣", "感兴趣"])
    print(f"推荐系统准确率：{accuracy:.4f}")
    print("详细分类报告：")
    print(report)


# 5. 主函数：加载、清洗、处理数据
def main():
    # 路径配置（示例文件路径，可根据需要调整）
    news_path = "news.tsv"  # 新闻数据文件路径
    behaviors_path = "behaviors.tsv"  # 用户行为数据文件路径
    # 加载数据
    news_df, behaviors_df = load_mind_dataset(news_path, behaviors_path)
    # 数据清洗
    news_df, behaviors_df = clean_data(news_df, behaviors_df)
    # 特征提取
    news_df, behaviors_df = extract_features(news_df, behaviors_df)
    # 数据编码与分割
    news_df, train_behaviors, test_behaviors = encode_and_split(
        news_df, behaviors_df)
    print("\n预处理完成，数据准备就绪！")

    # 模拟用户和物品数据
    user_descriptions, book_descriptions = create_sample_data()
    # 生成用户与物品嵌入
    user_embeddings, book_embeddings = generate_user_item_embeddings(
        user_descriptions, book_descriptions)
    # 计算相似度并推荐
    calculate_similarity(user_embeddings, book_embeddings,
                         user_descriptions, book_descriptions)

    model_name = "bert-base-chinese"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = PretrainedRecommendationModel(model_name).to(device)
    # 数据准备
    user_texts, item_texts, labels = create_sample_data()
    dataset = RecommendationDataset(user_texts, item_texts, labels, tokenizer)
    data_loader = DataLoader(dataset, batch_size=2, shuffle=True)
    # 模型训练与评估
    optimizer = optim.Adam(model.parameters(), lr=5e-5)
    criterion = nn.BCELoss()
    train_model(model, data_loader, optimizer, criterion, epochs=3)
    evaluate_model(model, data_loader)

    model_name = "bert-base-chinese"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = PretrainedRecommendationModel(model_name).to(device)
    # 加载训练好的权重（假设已经训练完成并保存）
    model.load_state_dict(torch.load(
        "pretrained_recommendation_model.pth"))
    # 创建推理数据集
    user_texts, item_texts, labels = create_sample_data()
    dataset = RecommendationDataset(user_texts, item_texts, labels, tokenizer)
    data_loader = DataLoader(dataset, batch_size=2)
    # 推理过程
    print("开始推理用户与物品的匹配结果...")
    predictions, true_labels = inference(model, data_loader)
    # 评估性能
    print("评估推荐系统性能...")
    evaluate(predictions, true_labels)
