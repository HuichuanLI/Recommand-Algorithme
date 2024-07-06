58xueke.com
import torch

# 检查 PyTorch 版本：确保输出的 PyTorch 版本是支持 GPU 的版本。
print(torch.__version__)
# 例如，版本号中包含 +cpu 表示 CPU 版本，而 +cu 表示 GPU 版本。
# 例如，1.9.0+cu102 表示支持 CUDA 10.2 的 GPU 版本。


# 检查 CUDA 版本：如果您的 PyTorch 版本支持 GPU，您还需要确认您的系统上安装了与 PyTorch 兼容的 CUDA 版本。
print(torch.version.cuda)
# 确保输出的 CUDA 版本与 PyTorch 所需的版本匹配。


# 检查 GPU 是否可用：确保 PyTorch 能够检测到可用的 GPU
print(torch.cuda.is_available())
# 如果输出为 True，则表示 PyTorch 可以使用 GPU。


# 检查 cuDNN 是否可用
print(torch.backends.cudnn.version())
# 如果输出一个版本号，那么 cuDNN 已经成功安装并且可以在 PyTorch 中使用。
# 如果输出 None，则表示 cuDNN 可能没有正确安装或没有与 PyTorch 版本兼容。


# 运行结果：
# 2.1.2+cpu
# None
# False
# None

# 安装GPU版本的torch，查询自己的电脑环境下载支持的版本：
# pip3 install torch torchvision torchaudio --index-url
# https://download.pytorch.org/whl/cu118

# 成功安装：
# Successfully installed torch-2.2.0+cu118 torchaudio-2.2.0+cu118
# torchvision-0.17.0+cu118

# 再次运行结果：
# 2.2.0+cu118
# 11.8
# True
# 8700
