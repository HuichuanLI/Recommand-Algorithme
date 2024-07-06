import json
import os
from ast import literal_eval

from utils import *
from db_qdrant import Qdrant


def preprocess_data(df_news):
    # 数据预处理块

    # 将包含字符串表示的列表转换为实际的列表
    # pd.notna(x) 检查x是否为非缺失值（即不是NaN），确保不对缺失值进行转换。
    # literal_eval(x) 是一个安全的方式来将字符串转换为相应的Python对象
    df_news['title_entities'] = df_news['title_entities'].apply(
        lambda x: literal_eval(x) if pd.notna(x) else [])
    df_news['abstract_entities'] = df_news['abstract_entities'].apply(
        lambda x: literal_eval(x) if pd.notna(x) else [])

    # 使用空字符串填充其他列的 NaN 值
    df_news = df_news.fillna('')

    # 新增 news_info 列，合并`类别、子类别、标题、摘要`字符串
    concatenation_order = ["category", "sub_category", "title", "abstract"]
    df_news['news_info'] = df_news.apply(lambda row: ' '.join(
        f"[{col}]:{row[col]}" for col in concatenation_order), axis=1)
    news_info_list = df_news['news_info'].values.tolist()
    logger.trace(
        f"新增 news_info 列 | len(news_info_list): {len(news_info_list)}")
    return df_news, news_info_list


def store_embeddings_to_json(embeddings, ids, payloads, file_path):
    # 存储嵌入为 JSON 文件
    json_data = {
        "batch_ids": ids,
        "batch_embeddings": embeddings,
        "batch_payloads": payloads
    }
    with open(file_path, 'w') as json_file:
        json.dump(json_data, json_file)


def compute_and_store_embeddings(data_list, embedding_folder, batch_size=1000):
    # 嵌入计算和存储块

    # 分批次向量化
    ids = list(range(1, len(data_list) + 1))  # 生成递增的 ids 列表

    for batch_idx, i in enumerate(range(0, len(data_list), batch_size)):
        # 获取批次数据 batch_ids、batch_payloads
        batch_ids = ids[i:i + batch_size]
        df_news_batch = df_news.iloc[i:i + batch_size]
        batch_payloads = df_news_batch.to_dict(orient='records')

        # 计算嵌入 batch_embeddings
        batch_data = data_list[i:i + batch_size]
        batch_embeddings = embed_sentences(batch_data)

        # 存储为 JSON 文件
        file_path = os.path.join(
            embedding_folder,
            f"batch_{batch_idx + 1}.json")
        store_embeddings_to_json(
            batch_embeddings,
            batch_ids,
            batch_payloads,
            file_path)

        # 打印存储信息
        logger.info(f"批次 {batch_idx} 数据存储成功，文件路径: {file_path}")


def load_embeddings_and_save_to_qdrant(
        collection_name,
        embedding_folder,
        batch_size):
    # 加载嵌入和存储到向量数据库

    qdrant = Qdrant()

    # 创建新的集合
    if qdrant.create_collection(collection_name):
        logger.success(f"创建集合成功 | collection_name: {collection_name}")
    else:
        logger.error(f"创建集合失败 | collection_name: {collection_name}")

    # 分批次存储到向量数据库
    for batch_idx, i in enumerate(range(0, len(news_info_list), batch_size)):
        # 读取 JSON 文件
        file_path = os.path.join(
            embedding_folder,
            f"batch_{batch_idx + 1}.json")
        if os.path.exists(file_path):
            with open(file_path, 'r') as json_file:
                json_data = json.load(json_file)

                batch_ids = json_data["batch_ids"]
                batch_embeddings = json_data["batch_embeddings"]
                batch_payloads = json_data["batch_payloads"]

                # 插入数据到 Qdrant
                if qdrant.add_points(
                        collection_name,
                        batch_ids,
                        batch_embeddings,
                        batch_payloads):
                    logger.success(f"批次 {batch_idx + 1} 插入成功")
                else:
                    logger.error(f"批次 {batch_idx + 1} 插入失败")
        else:
            logger.warning(f"文件 {file_path} 不存在，跳过该批次数据的插入。")

    logger.info("所有数据插入完成。")


# 读取新闻数据
df_news = get_df_news()

# 数据预处理
df_news, news_info_list = preprocess_data(df_news)

# 指定存储 embeddings 的文件夹路径
embedding_folder = 'embeddings_folder'
os.makedirs(embedding_folder, exist_ok=True)

# 计算和存储嵌入
compute_and_store_embeddings(news_info_list, embedding_folder, 1000)

# 加载嵌入和存储到向量数据库
collection_name = "all_news"
# load_embeddings_and_save_to_qdrant(collection_name, embedding_folder, 1000)
