# -*- coding: utf-8 -*-

import copy
import threading

import pymysql
from dbutils.pooled_db import PooledDB

from .. import logger
from ..config import mysql_config


class DB(object):
    lock = threading.Lock()
    _pool = None  # 数据库连接池对象
    instance = None  # 全局通用的DB对象

    def __init__(self):
        self.pool = DB._get_conn_pool()

    @staticmethod
    def _get_conn_pool():
        if DB._pool is None:
            DB.lock.acquire()  # 获的锁
            try:
                if DB._pool is None:
                    cfg = copy.deepcopy(mysql_config.cfg)
                    creator = eval(cfg['creator'])  # 具体执行引擎
                    del cfg['creator']
                    DB._pool = PooledDB(creator, **cfg)
            finally:
                DB.lock.release()  # 释放锁
            pass
        return DB._pool

    @staticmethod
    def get_instance():
        if DB.instance is None:
            DB.instance = DB()
        return DB.instance

    def _get_connection(self):
        conn = self.pool.connection()
        cursor = conn.cursor(cursor=pymysql.cursors.DictCursor)
        return conn, cursor

    @staticmethod
    def _close_connection(conn, cursor):
        if cursor:
            try:
                cursor.close()
            except Exception as e:
                logger.error("关闭Mysql连接cursor异常。", exc_info=e)
        if conn:
            try:
                conn.close()
            except Exception as e:
                logger.error("关闭Mysql连接conn异常。", exc_info=e)

    # 对外的API
    @staticmethod
    def query_sql(sql, **params):
        instance = DB.get_instance()
        conn, cursor = instance._get_connection()
        try:
            cursor.execute(sql, params)
            result = cursor.fetchall()
            return result
        except Exception as e:
            raise ValueError(f"Query查询异常，当前sql语句为:{sql}, 参数信息为:{params}") from e
        finally:
            instance._close_connection(conn, cursor)
