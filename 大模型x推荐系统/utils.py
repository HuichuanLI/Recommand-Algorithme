import sys
from collections import Counter

import pandas as pd
import torch
from loguru import logger

from NewsGPT import NewsGPT
from sentence_transformers import SentenceTransformer, util

logger.remove()  # 删去import logger之后自动产生的handler，不删除的话会出现重复输出的现象
logger.add(sys.stderr, level="DEBUG")  # 调整日志输出级别: INFO|DEBUG|TRACE


model_name = 'Dmeta-embedding'
device = "cuda" if torch.cuda.is_available() else "cpu"
logger.info("Use pytorch device: {}".format(device))
model = SentenceTransformer(model_name, device=device)


def embed_sentences(sentences):
    """
    使用预训练的 SentenceTransformer 模型对句子列表进行嵌入。

    参数：
    - sentences：要嵌入的句子列表。
    - model_name：SentenceTransformer 模型的名称。

    返回：
    - embeddings：句子嵌入的列表。
    """
    embeddings = model.encode(sentences, normalize_embeddings=True)
    embeddings = embeddings.tolist()
    logger.trace(f"向量化 | embeddings: {len(embeddings)}, {type(embeddings)}")
    return embeddings


def get_df_behaviors_sample(sample_size=10):

    # 读取行为数据
    df_behaviors = pd.read_csv(
        'MIND/MINDsmall_train/behaviors.tsv',
        names=[
            "impression_id",
            "user_id",
            "time",
            "click_history",
            "impression_lpg"],
        sep='\t',
        header=None)

    # 采样
    df_behaviors_sample = df_behaviors.head(sample_size)
    logger.info(
        f"采样后 | df_behaviors_sample.shape: {df_behaviors_sample.shape}")

    return df_behaviors_sample


def get_df_news():

    # 读取新闻数据
    df_news = pd.read_csv(
        'MIND/MINDsmall_train/news.tsv',
        names=[
            "news_id",
            "category",
            "sub_category",
            "title",
            "abstract",
            "url",
            "title_entities",
            "abstract_entities"],
        sep='\t',
        header=None)
    logger.info(f"df_news.shape: {df_news.shape}")

    return df_news


def get_user_favorite_combinations(click_history, df_news):

    # 从 df_news 中查询并构建用户最喜欢的（category, sub_category）组合
    user_favorite_combinations = Counter()
    for news_id in click_history:
        category = df_news.loc[df_news['news_id']
                               == news_id, 'category'].values[0]
        sub_category = df_news.loc[df_news['news_id']
                                   == news_id, 'sub_category'].values[0]
        user_favorite_combinations.update([(category, sub_category)])

    logger.info(
        f"统计用户偏好类别组合 | user_favorite_combinations: {user_favorite_combinations}")
    return user_favorite_combinations


def generate_historical_records(df_news, click_history):
    # 根据 click_history 查询每条新闻的详细信息，并组合成字符串
    historical_records = []
    for idx, news_id in enumerate(click_history, start=1):
        # 查询每条新闻的详细信息
        record = df_news[df_news['news_id'] == news_id][[
            "category", "sub_category", "title", "abstract"]]
        # 组合成字符串，添加序号
        record_str = f"{idx}. " + ' '.join(
            f"[{col}]:{record.iloc[0][col]}" for col in [
                "category", "sub_category", "title", "abstract"])
        historical_records.append(record_str)
    logger.trace(f"历史交互 | historical_records: {historical_records}")
    return historical_records


def generate_user_profile(historical_records_str):
    # 生成用户画像: 通过理解`用户历史行为序列`，生成`用户感兴趣的话题`以及`用户位置信息`
    prompt = f"""Describe user profile based on browsed news list in the following format:

{historical_records_str}

You should describe the related topics and regions in the following format:

[topics]
-[topic1]

[region]
-[region1]
"""
    logger.info(f"prompt: \n{prompt}")

    # 模拟 NewsGPT 的调用
    gpt = NewsGPT()
    user_profile = gpt.get_completion(prompt)
#     user_profile = '''
# [topics]
# -TV Entertainment
# -Sports
# -Crime
# -Lifestyle
# -Movies
# -Politics
# [regions]
# -United States'''
    logger.success(f"用户画像 | user_profile: \n{user_profile}")
    return user_profile


def fine_ranking(
        user_profile,
        historical_records_str,
        candidate_list,
        top_n=5):
    prompt = f"""
I want you to recommend news to a user based on some personal information and historical records of news reading.

User profile: ```
{user_profile}
```

The historical records include news category, subcategory, title, and abstract. You are encouraged to learn his news preferences from the news he has read. Here are some examples:```
{historical_records_str}
```

Here's a list of news that he is likely to like: ```
{candidate_list}
```

Please select the top {top_n} news in the list that is most likely to be liked.
Please only output the order of these news in the form of a numerical list.
Example Output Format: 1,8,2,12,7

Output: """
    logger.info(f"prompt: \n{prompt}")
    gpt = NewsGPT()
    response = gpt.get_completion(prompt)
    logger.success(f"response: \n{response}")

    top_news = response.strip().split(',')
    return top_news
