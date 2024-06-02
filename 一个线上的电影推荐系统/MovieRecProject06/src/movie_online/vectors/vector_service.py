# -*- coding: utf-8 -*-
import threading
import time
from typing import Dict

import faiss
import numpy as np

from .. import logger
from ..utils.mysql_util import DB
from ..services.spu_feature_service import SpuFeatureService
from ..models.model_service import ModelService


class FaissEntity:
    def __init__(self, name, save_info_path, measure=faiss.METRIC_INNER_PRODUCT, param="HNSW4"):
        self.save_info_path = save_info_path  # 数据保存路径
        self.lock = threading.Lock()  # 锁
        self.name = name  # 索引名称，eg: "dssm_202211061017"
        self.measure = measure  # 索引的相似度度量方式
        self.param = param  # 具体的索引类型，比如: HNSW64
        self.is_trained = False  # 索引的faiss模型是否原本就训练好了，如果训练好的话，就支持线上新商品向量累加/追加

        save_infos = np.load(save_info_path, allow_pickle=True)
        self.embedding = np.asarray(save_infos['spu_embedding']).astype('float32')  # 数据加载
        self.id_mapping = dict(save_infos['id_mapping'])  # id映射文件 内部id/行id ---> 实际商品id
        self.idx_to_inner_idx_mapping = self.build_inverse_mapping()  # 实际商品id --> 内部id/行id
        self.dim = len(self.embedding[0])  # 向量的维度大小
        self._size = len(self.id_mapping)  # 商品的数量
        self.index = None  # faiss里面的索引对象

    def save(self):
        np.savez_compressed(
            self.save_info_path,
            spu_embedding=self.embedding,  # embedding信息
            id_mapping=np.asarray(list(self.id_mapping.items()))  # id映射
        )

    def build_inverse_mapping(self):
        result = {}
        for inner_idx, idx in self.id_mapping.items():
            if idx not in result:
                result[idx] = inner_idx
            else:
                result[idx] = max(result[idx], inner_idx)  # 保存最新的内部id映射
        return result

    def add_vector(self, _id, _vector):
        if _id is None or _vector is None:
            raise ValueError("id和vector不允许为空!")
        if not self.is_trained:
            raise ValueError(f"当前索引不支持追加，索引类型为:{self.param}")
        _vector = np.asarray(_vector, dtype='float32')
        _vector = _vector.reshape((1, -1))
        if len(_vector[0]) != self.dim:
            raise ValueError(f"索引内部向量维度大小为:{self.dim}, 传入向量维度大小为:{len(_vector)}")
        self.lock.acquire()  # 获取锁
        try:
            # if _id in self.idx_to_inner_idx_mapping:
            #     logger.info("商品id已经存在于索引中，不进行追加操作!!")
            # else:
            #     self.id_mapping[self._size] = _id  # 添加id映射
            #     self.idx_to_inner_idx_mapping = self.build_inverse_mapping()
            #     self.embedding = np.vstack([self.embedding, _vector])  # 向量添加
            #     self._size = self._size + 1  # 商品数量增加
            #     self.index.add(_vector)  # 加入到索引中
            #     # 输出到文件(保证下次恢复的时候存在)
            #     self.save()

            self.id_mapping[self._size] = _id  # 添加id映射
            self.idx_to_inner_idx_mapping = self.build_inverse_mapping()
            self.embedding = np.vstack([self.embedding, _vector])  # 向量添加
            self._size = self._size + 1  # 商品数量增加
            self.index.add(_vector)  # 加入到索引中
            # 输出到文件(保证下次恢复的时候存在)
            self.save()
        finally:
            self.lock.release()

    def search(self, _vector, _k):
        _vector = np.asarray(_vector, dtype='float32')
        _vector = _vector.reshape((1, -1))
        if len(_vector[0]) != self.dim:
            raise ValueError(f"索引内部向量维度大小为:{self.dim}, 传入向量维度大小为:{len(_vector)}")
        r = self.index.search(_vector, _k)  # 直接调用faiss的index的API
        return [int(self.id_mapping[_r]) for _r in r[1][0] if _r in self.id_mapping]  # 将索引向量对应的行号转换为实际的商品id

    def search_idx(self, _idx, _k):
        # 1. 获取当前商品id对应的向量
        _inner_idx = self.idx_to_inner_idx_mapping.get(_idx)
        if _inner_idx is None:
            raise ValueError(f"未找到当前商品对应的特征向量:{_idx}")
        _vector = self.embedding[_inner_idx].reshape((1, -1))
        if len(_vector[0]) != self.dim:
            raise ValueError(f"索引内部向量维度大小为:{self.dim}, 传入向量维度大小为:{len(_vector)}")
        r = self.index.search(_vector, _k)  # 直接调用faiss的index的API
        # 遍历处理 将索引向量对应的行号转换为实际的商品id
        result = []
        for _r in r[1][0]:
            if (_r in self.id_mapping) and (_r != _inner_idx):
                result.append(self.id_mapping[_r])
        return result

    def destroy(self):
        self.lock.acquire()
        try:
            self.index.reset()  # 索引的销毁(减少内存的占用)
            self.id_mapping = {}
            self.embedding = {}
        finally:
            self.lock.release()


# noinspection SqlDialectInspection,SqlNoDataSourceInspection
class _InnerVectorService:
    def __init__(self):
        self.fm_spu_model = None
        self.dssm_spu_model = None
        self.index_mapping: Dict[str, FaissEntity] = {}  # 索引名称和index对象的映射
        # 初始化恢复
        try:
            datas = DB.query_sql(
                sql="select * from vector_info where state=0"
            )
            for data in datas:
                entity = FaissEntity(
                    name=data['name'],
                    save_info_path=data['embedding_path'],
                    measure=int(data['measure']),
                    param=data['param']
                )
                # 插入数据
                self.build_faiss_index(entity)
            logger.info(f"初始化所有向量库完成，总恢复索引数量:{len(datas)}")
        except Exception as e:
            raise ValueError("初始化向量库失败。") from e

    def build_faiss_index(self, entity: FaissEntity):
        if entity.name in self.index_mapping:
            raise ValueError(f"{entity.name}名称的索引存在，不允许重复创建，请更改名称或者修改后再创建!")
        start_time = time.time()
        index = faiss.index_factory(entity.dim, entity.param, entity.measure)
        entity.is_trained = index.is_trained  # 原始标注是否已经训练好
        if not index.is_trained:
            # 训练
            index.train(entity.embedding)
        # 插入添加
        index.add(entity.embedding)
        entity.index = index  # 参数更新
        self.index_mapping[entity.name] = entity  # 映射表更新
        end_time = time.time()
        logger.info(f"创建faiss索引完成:{entity.name}, 向量数:{entity._size}, 耗时:{end_time - start_time}s")

    # def get_fm_spu_model(self):
    #     if self.fm_spu_model is None:
    #         self.fm_spu_model = fm_spu_side.AlgorithmicModel(
    #             root_dir=os.path.join(global_config.model_root_dir, "fm")
    #         )
    #     return self.fm_spu_model

    # def get_dssm_spu_model(self):
    #     if self.dssm_spu_model is None:
    #         self.dssm_spu_model = dssm_spu_side.AlgorithmicModel(
    #             root_dir=os.path.join(global_config.model_root_dir, "dssm")
    #         )
    #     return self.dssm_spu_model


# noinspection SqlDialectInspection,SqlNoDataSourceInspection,PyUnresolvedReferences
class VectorService:
    proxy = _InnerVectorService()  # 可以保证proxy仅初始化一次 ---> 单例

    @staticmethod
    def get_measure(measure: str):
        """
        基于参数字符串，获取对应的向量相似度度量方式
        :param measure:
        :return:
        """
        if measure == 'inner':
            # 余弦相似度 --> 前提条件是向量做了norm-2的转换
            return faiss.METRIC_INNER_PRODUCT
        elif measure == 'l1':
            return faiss.METRIC_L1
        else:
            return faiss.METRIC_L2

    @staticmethod
    def build_faiss_index(entity: FaissEntity):
        """
        构建或者恢复向量索引
        :param entity:
        :return:
        """
        if entity.name in VectorService.proxy.index_mapping:
            raise ValueError(f"{entity.name}名称的索引存在，不允许重复创建，请更改名称或者修改后再创建!")
        # 查看数据库中该索引是否存在
        d = DB.query_sql(
            sql="select * from vector_info where name=%(name)s limit 1",
            name=entity.name
        )
        if len(d) > 0:
            d = d[0]
            # 如果存在数据，那么直接恢复
            entity = FaissEntity(
                name=d['name'],
                save_info_path=d['save_info_path'],
                measure=int(d['measure']),
                param=d['param']
            )
            # 更新数据库
            d = DB.update_sql(
                sql="update vector_info set state=0 where name=%(name)s",
                name=entity.name
            )
            if d <= 0:
                raise ValueError("更新数据库失败!")
        else:
            # 插入数据库
            d = DB.execute_sql(
                sql="""
                insert into vector_info(`name`,`embedding_path`,`id_mapping_path`,`measure`,`param`)
                values (%(name)s, %(save_info_path)s, %(save_info_path)s, %(measure)s, %(param)s)
                """,
                name=entity.name,
                save_info_path=entity.save_info_path,
                measure=entity.measure,
                param=entity.param
            )
            print(d)
            if d <= 0:
                raise ValueError("往数据库插入数据失败!")
        try:
            # 实际插入（服务内部实际构建索引）
            VectorService.proxy.build_faiss_index(entity)
        except Exception as e:
            # build失败的情况下，需要更新数据库
            DB.update_sql(
                sql="update vector_info set state=1 where name=%(name)s",
                name=entity.name
            )
            raise e

    @staticmethod
    def delete_index(name):
        """
        删除索引：数据库标记位设置为1、当前服务删除
        :param name:
        :return:
        """
        if name not in VectorService.proxy.index_mapping:
            return
        # 对象销毁
        VectorService.proxy.index_mapping[name].destroy()
        del VectorService.proxy.index_mapping[name]
        # 更新数据库
        DB.update_sql(
            sql="update vector_info set state=1 where name=%(name)s",
            name=name
        )

    @staticmethod
    def list_index():
        return {k: v._size for k, v in VectorService.proxy.index_mapping.items()}

    @staticmethod
    def get_index(name):
        if name not in VectorService.proxy.index_mapping:
            raise ValueError(f"faiss索引{name}不存在!")
        return VectorService.proxy.index_mapping[name]

    @staticmethod
    def search(name, vector, k):
        if name not in VectorService.proxy.index_mapping:
            print(name)
            raise ValueError(f"faiss索引{name}不存在!")
        return VectorService.proxy.index_mapping[name].search(vector, k)

    @staticmethod
    def search_by_id(name, idx, k):
        if name not in VectorService.proxy.index_mapping:
            raise ValueError(f"faiss索引{name}不存在!")
        return VectorService.proxy.index_mapping[name].search_idx(idx, k)

    @staticmethod
    def add_vector(name, model_version, spu_id):
        spu = SpuFeatureService.get_effect_spu_features([spu_id]).get(spu_id)
        if spu is None:
            raise ValueError(f"当前商品不存在:{spu_id}")

        # 调用物品侧模型获取对应的物品向量
        vector_result = None
        if name == 'fm':
            vector_result = ModelService.fetch_predict_result(
                model_register_name="fm_spu",
                model_version=model_version,
                spu=spu
            )
        elif name == 'dssm':
            vector_result = None
            # version, vector = VectorService.proxy.get_dssm_spu_model().fetch_spu_vector(spu=None, spu_id=spu_id)
        else:
            raise ValueError(f"当前仅支持fm和dssm向量模型:{name}")
        if vector_result is None:
            raise ValueError(f"当前获取向量失败:{name} - {model_version} - {spu_id}")
        vector = vector_result[1]  # 模型返回结果，也就是向量
        version = vector_result[0]['version']  # 获取得到当前的模型/索引版本字符串
        # 开始添加
        name = version
        if name not in VectorService.proxy.index_mapping:
            raise ValueError(f"当前index索引未加载，无法追加特征，索引名称为:{name}")
        VectorService.proxy.index_mapping[name].add_vector(spu_id, vector)
        return name
