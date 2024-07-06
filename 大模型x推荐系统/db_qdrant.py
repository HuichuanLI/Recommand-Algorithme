from loguru import logger
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, Batch
from qdrant_client.http.exceptions import UnexpectedResponse  # 捕获错误信息

from config import QDRANT_HOST, QDRANT_PORT, QDRANT_EMBEDDING_DIMS


class Qdrant:
    def __init__(self):
        self.client = QdrantClient(
            host=QDRANT_HOST,
            port=QDRANT_PORT)  # 创建客户端实例
        self.size = QDRANT_EMBEDDING_DIMS  # openai embedding 维度 = 1536

    def get_points_count(self, collection_name):
        """
        先检查集合是否存在。
        - 如果集合存在，返回该集合的 points_count （集合中确切的points_count）
        - 如果集合不存在，创建集合。
            - 创建集合成功，则返回 points_count （0: 刚创建完points_count就是0）
            - 创建集合失败，则返回 points_count （-1: 创建失败了，定义points_count为-1）

        Returns:
            points_count

        Raises:
            UnexpectedResponse: 如果在获取集合信息时发生意外的响应。
            ValueError: Collection test_collection not found
        """
        try:
            collection_info = self.get_collection(collection_name)
        except (UnexpectedResponse, ValueError) as e:  # 集合不存在，创建新的集合
            if self.create_collection(collection_name):
                logger.success(
                    f"创建集合成功 | collection_name: {collection_name} points_count: 0")
                return 0
            else:
                logger.error(
                    f"创建集合失败 | collection_name: {collection_name} 错误信息:{e}")
                return -1
        except Exception as e:
            logger.error(
                f"获取集合信息时发生错误 | collection_name: {collection_name} 错误信息:{e}")
            return -1  # 返回错误码或其他适当的值
        else:
            points_count = collection_info.points_count
            logger.success(
                f"库里已有该集合 | collection_name: {collection_name} points_count：{points_count}")
            return points_count

    def list_all_collection_names(self):
        """
        CollectionsResponse类型举例：
        CollectionsResponse(collections=[
            CollectionDescription(name='GreedyAIEmployeeHandbook'),
            CollectionDescription(name='python')
        ])
        CollectionsResponse(collections=[])
        """
        CollectionsResponse = self.client.get_collections()
        collection_names = [
            CollectionDescription.name for CollectionDescription in CollectionsResponse.collections]
        return collection_names

    # 获取集合信息
    def get_collection(self, collection_name):
        """
        获取集合信息。

        Args:
            collection_name (str, optional): 自定义的集合名称。如果未提供，则使用默认的self.collection_name。

        Returns:
            collection_info: 集合信息。
        """
        collection_info = self.client.get_collection(
            collection_name=collection_name)
        return collection_info

    # 创建集合
    def create_collection(self, collection_name) -> bool:
        """
        创建集合。

        Args:
            collection_name (str, optional): 自定义的集合名称。如果未提供，则使用默认的self.collection_name。

        Returns:
            bool: 如果成功创建集合，则返回True；否则返回False。
        """

        return self.client.recreate_collection(
            collection_name=collection_name, vectors_config=VectorParams(
                size=self.size, distance=Distance.COSINE), )

    def add_points(self, collection_name, ids, vectors, payloads):
        # 将数据点添加到Qdrant
        self.client.upsert(
            collection_name=collection_name,
            wait=True,
            points=Batch(
                ids=ids,
                payloads=payloads,
                vectors=vectors
            )
        )
        return True

    # 搜索
    def search(self, collection_name, query_vector, limit=3):
        return self.client.search(
            collection_name=collection_name,
            query_vector=query_vector,
            limit=limit,
            with_payload=True
        )

    def search_with_query_filter(
            self,
            collection_name,
            query_vector,
            query_filter,
            limit=3):
        """
        根据向量相似度和指定的过滤条件，在集合中搜索最相似的points。
        API 文档：https://qdrant.github.io/qdrant/redoc/index.html#tag/points/operation/search_points
        :param collection_name:要搜索的集合名称
        :param query_vector:用于相似性比较的向量
        :param query_filter:过滤条件
        :param limit:要返回的结果的最大数量
        :return:
        """
        return self.client.search(
            collection_name=collection_name,
            query_filter=query_filter,
            query_vector=query_vector,
            limit=limit,
            with_payload=True
        )


if __name__ == "__main__":
    qdrant = Qdrant()

    # 创建集合
    # collection_name = "test"

    # 获取集合信息
    # qdrant.get_collection(collection_name)
    # 如果之前没有创建集合，则会报以下错误
    # qdrant_client.http.exceptions.UnexpectedResponse: Unexpected Response: 404 (Not Found)
    # Raw response content:
    # b'{"status":{"error":"Not found: Collection `test` doesn\'t exist!"},"time":0.000198585}'

    # 获取集合信息，如果没有该集合则创建
    collection_name = "all_news"
    count = qdrant.get_points_count(collection_name)
    print(count)
    # 如果之前没有创建集合，且正确创建了该集合，则输出0。例：创建集合成功。集合名：test。节点数量：0。
    # 如果之前创建了该集合，则输出该集合内部的节点数量。例：库里已有该集合。集合名：test。节点数量：0。

    # 删除集合
    # collection_name = "test"
    # qdrant.client.delete_collection(collection_name)
