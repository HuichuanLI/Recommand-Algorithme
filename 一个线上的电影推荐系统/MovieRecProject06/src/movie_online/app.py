# -*- coding: utf-8 -*-
import os
import shutil

import flask
from flask import Flask, request, jsonify

from . import logger
from .config import global_config
from .entity import RecItem
from .entity.config_param import ConfigParams
from .entity.scene_meta import SceneMeta
from .entity.user_feature import UserFeatureEntity
from .models.model_service import ModelService
from .strategy.strategy_runner import StrategyRunner

# noinspection PyProtectedMember
from .vectors.vector_service import VectorService, FaissEntity


def json_default(o):
    if isinstance(o, RecItem):
        return o.to_dict()
    else:
        return flask.json.provider._default(o)


app = Flask(__name__)
app.json.ensure_ascii = False  # 当前flask版本有效，给定json格式数据返回的时候，针对中文不进行编码处理
# 给定返回结果的时候，对象如何转换为json字符串；默认情况下，自定义对象是无法转换的；不同flask版本，最终解决代码可能不一样
app.json.default = json_default  # 2.3.3 flask

runner = StrategyRunner()


@app.route("/")
@app.route("/index")
def index():
    return "简易推荐系统后端接口"


# region 构建开发过程中的测试api

@app.route("/test1")
def test1():
    try:
        # 1. 获取参数并进行参数的check
        args = request.args
        name = args.get('name')
        age = args.get('age')

        # 2. 调用具体的逻辑代码，执行并返回结果

        # 3. 执行的结果拼接，返回最终值
        return jsonify({
            'code': 200,
            'data': {
                'name': name,
                'age': age
            }
        })
    except Exception as e:
        logger.error("服务器接口执行异常.", exc_info=e)
        return jsonify({
            'code': 201,
            'msg': f'服务器接口执行异常:{e}'
        })


@app.route("/test2")
def test2():
    scene: SceneMeta = SceneMeta()
    config: ConfigParams = ConfigParams(
        number_push=500,
        user=UserFeatureEntity(
            record={
                'user_id': 196,
                'zip_code': '403205'
            }
        )
    )
    result = runner.get_rec_items_by_scene(scene, config)
    return jsonify({'code': 200, 'data': result})


# endregion


# region 文件管理相关逻辑API: 上传、删除等


# noinspection DuplicatedCode
@app.route("/uploader", methods=['POST'])
def uploader():
    """
    上传文件到服务器上
    :return:
    """
    _args = flask.request.values
    for c in ['name', 'version']:
        if c not in _args:
            return flask.jsonify({"code": 201, "msg": f"必须给定{c}参数!"})
    name = _args.get("name")
    version = _args.get('version')
    sub_dirs = _args.get('sub_dir_names', '')  # 子文件夹的名称字符串列表，使用","分割开的一个字符串
    sub_dirs = [sub_dir.strip() for sub_dir in sub_dirs.split(",")]
    filename = _args.get("filename")  # 上传文件新的文件文件名称
    file = flask.request.files['file']  # 获取待上传的对象
    if filename is None:
        # 当没有给定上传文件重名称的时候，直接使用上传文件的原本名称
        filename = file.filename
    _dir = os.path.join(global_config.model_root_dir, name, version, *sub_dirs)
    os.makedirs(_dir, exist_ok=True)  # 创建输出的文件夹
    save_path = os.path.join(_dir, filename)
    if os.path.exists(save_path):
        return flask.jsonify({
            "code": 202,
            "msg": f"文件路径已存在，请重新给定name、version和filename参数值，当前参数为:{name} -- {version} -- {filename}"
        })
    file.save(os.path.abspath(save_path))  # 保存操作
    return flask.jsonify({"code": 200, "name": name, "version": version, "filename": filename})


# noinspection DuplicatedCode
@app.route("/deleter", methods=['GET'])
def deleter():
    """
    删除服务器上的文件
    eg: s.get("http://127.0.0.1:9999/deleter", params={"version":version, "name":name})
    最终删除的文件是:
    {global_config.model_root_dir}/{name}/{version}
    :return:
    """
    _args = flask.request.values
    for c in ['name', 'version']:
        if c not in _args:
            return flask.jsonify({"code": 201, "msg": f"必须给定{c}参数!"})
    name = _args.get("name")
    version = _args.get('version')
    sub_dirs = _args.get('sub_dir_names', '')  # 子文件夹的名称字符串列表，使用","分割开的一个字符串
    sub_dirs = [sub_dir.strip() for sub_dir in sub_dirs.split(",")]

    _dir = os.path.join(global_config.model_root_dir, name, version, *sub_dirs)

    filename = _args.get("filename")
    if filename is None:
        # 删除文件夹
        _file = _dir
    else:
        # 删除的具体filename文件
        _file = os.path.join(_dir, filename)
    shutil.rmtree(_file)  # 删除操作
    return flask.jsonify({"code": 200, "msg": f"文件删除成功:{_file}"})


# endregion

# region 模型管理服务相关API


@app.route('/model/list', methods=['GET'])
def list_model():
    """
    展示当前模型服务内有多少个模型
    :return:
    """
    return flask.jsonify({"code": 200, "data": ModelService.list_models()})


# endregion

# region 索引服务相关API


@app.route('/faiss/list', methods=['GET'])
def list_faiss():
    """
    展示当前向量服务内部存在多少个索引
    :return:
    """
    return flask.jsonify({"code": 200, "data": VectorService.list_index()})


@app.route('/faiss/build', methods=['GET'])
def build_faiss():
    """
    索引构建，主要基于两个文件：向量矩阵文件、id映射文件；索引的必要参数：名称、类型、对应的构建参数
    :return:
    """
    try:
        # 1. 检查参数
        _args = flask.request.args
        for c in ['name', 'version']:
            if c not in _args:
                return flask.jsonify({"code": 201, "msg": f"必须给定{c}参数!"})
        # 2. 参数提取&检查
        name = _args.get('name')
        version = _args.get('version')
        save_info_file = _args.get('save_info_file', 'spu_embedding.npz')
        measure = VectorService.get_measure(_args.get('measure', 'inner'))
        param = _args.get('param', 'HNSW4')

        _dir = os.path.join(global_config.model_root_dir, name, version)
        save_info_path = os.path.join(_dir, save_info_file)
        if not os.path.exists(save_info_path):
            return flask.jsonify({"code": 201, "msg": f"必须给定save文件不存在{save_info_path}!"})
        # 3. 构建索引
        _entity = FaissEntity(
            name=f"{version}",
            save_info_path=save_info_path,
            measure=measure,
            param=param
        )
        VectorService.build_faiss_index(_entity)
        return flask.jsonify({"code": 200, "name": f"{name}_{version}", "msg": "索引构建成功!"})
    except Exception as e:
        logger.error("创建faiss索引失败.", exc_info=e)
        return flask.jsonify({"code": 202, "msg": f"创建faiss索引服务器异常:{e}"})


@app.route('/faiss/add', methods=['GET'])
def add_faiss():
    """
    增加一个商品的向量信息(新商品增加)
    :return:
    """
    try:
        # 1. 检查参数
        _args = flask.request.args
        for c in ['name', 'id']:
            if c not in _args:
                return flask.jsonify({"code": 201, "msg": f"必须给定{c}参数!"})
        # 2. 参数提取&检查
        name = _args.get('name')  # 模型名称
        version = _args.get('version')  # 模型版本字符串
        spu_id = int(_args.get('id'))  # 新商品id
        # 3. 索引内部增加商品向量
        name = VectorService.add_vector(name, version, spu_id)
        return flask.jsonify({"code": 200, "name": name, "version": version, "spu_id": spu_id, "msg": "增加向量成功!"})
    except Exception as e:
        logger.error("增加faiss向量失败.", exc_info=e)
        return flask.jsonify({"code": 202, "msg": f"增加faiss向量失败:{e}"})


@app.route('/faiss/delete', methods=['GET'])
def delete_faiss():
    """
    删除索引，删除后，索引就不可用了
    :return:
    """
    _args = flask.request.args
    for c in ['name', 'version']:
        if c not in _args:
            return flask.jsonify({"code": 201, "msg": f"必须给定{c}参数!"})
    # 2. 参数提取&检查
    name = _args.get('name')
    version = _args.get('version')
    # 3. 删除
    VectorService.delete_index(version)
    return flask.jsonify({"code": 200, "name": f"{name} - {version}", "msg": "索引删除成功!"})

# endregion
