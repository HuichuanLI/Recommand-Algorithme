import flask

from .entity.user_feature import UserFeatureEntity
from .strategy.entity.config_param import ConfigParams
from .strategy.entity.scene_meta import SceneMeta
from .strategy.strategy_runner import StrategyRunner


class JSONEncoder(flask.json.JSONEncoder):
    """
    主要目的是为了保证自定义的类型可以通过jsonify转换为json字符串返回给调用方
    NOTE: 要求目标对象o可以转换为dict字典
    """

    def default(self, o):
        return dict(o)


class Flask(flask.Flask):
    """
    指定默认的json_encoder，就是为了保证json格式的正常
    """
    json_encoder = JSONEncoder


app = Flask(import_name=__name__)
# 定义的全局策略入口
runner = StrategyRunner()


@app.route('/')
@app.route('/index')
def hello_world():
    return """
    <h1>欢迎进入我的推荐系统</h1>
    该系统主要提供一些推荐接口方便提取推荐结果
    """


@app.route("/f1", methods=['POST', 'GET'])
def f1():
    _config = ConfigParams()
    _scene = SceneMeta()
    _user = UserFeatureEntity(
        user_id=10001,  # 一般是前端传递过来
        location_id=27  # 一般根据用户id从数据库提取或者解析IP地址
    )
    # 调用预测
    _rs = runner.get_rec_spu_ids_by_scene(
        config=_config,
        scene=_scene,
        user=_user,
        spu=None
    )
    _data = {
        'code': 0,
        'data': _rs
    }
    return flask.jsonify(_data)
