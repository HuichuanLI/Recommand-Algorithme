58xueke.com
from utils import *
from db_qdrant import Qdrant
from qdrant_client.http import models

# 获取数据
df_behaviors_sample = get_df_behaviors_sample()
df_news = get_df_news()

# 循环 df_behaviors_sample 的每一行
for _, row in df_behaviors_sample.iterrows():
    user_id = row['user_id']
    click_history = row['click_history'].split()

    # 召回

    # 生成历史交互字符串 historical_records_str
    historical_records = generate_historical_records(df_news, click_history)
    historical_records_str = '\n'.join(historical_records)
    logger.info(
        f"历史交互字符串 | historical_records_str: \n{historical_records_str}")

    # 生成用户画像 user_profile
    user_profile = generate_user_profile(historical_records_str)

    # 向量化用户画像 user_emb
    user_emb = embed_sentences([user_profile])[0]

    # 过滤条件 query_filter
    # 统计出当前用户的（新闻类别，新闻子类别）偏好组合
    user_favorite_combinations = get_user_favorite_combinations(
        click_history, df_news)

    should_items = []
    for category, sub_category in user_favorite_combinations:
        should_item = models.Filter(
            must=[
                models.FieldCondition(
                    key="category",
                    match=models.MatchValue(
                        value=category,
                    )
                ),
                models.FieldCondition(
                    key="sub_category",
                    match=models.MatchValue(
                        value=sub_category,
                    )
                )
            ]
        )

        should_items.append(should_item)

    query_filter = models.Filter(
        should=should_items
    )

    # 使用 Qdrant 查询与用户画像字符串最相似的 news 列表
    qdrant = Qdrant()
    scored_point_list = qdrant.search_with_query_filter(
        "all_news", user_emb, query_filter, 20)
    coarse_top_news = [
        scored_point.payload for scored_point in scored_point_list]
    logger.info(f"len(top_news): {len(coarse_top_news)}")

    if coarse_top_news:
        # 排序
        coarse_top_news_str = '\n'.join(
            [f"{idx}. {news}" for idx, news in enumerate(coarse_top_news)])
        fine_top_news = fine_ranking(
            user_profile,
            historical_records_str,
            coarse_top_news_str,
            5)

        for idx in fine_top_news:
            news = coarse_top_news[int(idx)]
            logger.success(int(idx))
            logger.success(news)
    break
