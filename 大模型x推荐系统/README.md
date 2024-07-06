```58xueke.com
```
# 运行
1、创建虚拟环境并安装依赖
   - conda create -n py38_NewsGPT python=3.8
   - conda activate py38_NewsGPT
   - pip install -r requirements.txt

2、配置环境变量
   - 打开`.env.example`文件
   - 填写完整该文件中的`OPENAI_API_KEY`、`HTTP_PROXY`、`HTTPS_PROXY`三个环境变量
   - 把`.env.example`文件重命名为`.env`

3、docker运行`qdtant`容器

4、运行`save_news_to_qdrant.py`文件

5、运行`main.py`文件

## 注意
在安装`sentence-transformers`包时，会自动安装cpu版本的pytorch，可以运行embedding模型，但做批量embedding的时候会很慢。

如果你电脑有GPU，那需要1、卸载掉cpu版本的pytorch，2、安装gpu版本的pytorch。

可以运行`tests`文件下的`test_gpu.py`查看本机环境。
