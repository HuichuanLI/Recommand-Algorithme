# 调用100层 Tiramisu 进行图像分割
# Tiramisu 论文：https://arxiv.org/pdf/1611.09326.pdf
# API申请地址：https://www.remove.bg/
from removebg import RemoveBg
import os

# 引号内是你获取的API
rmbg = RemoveBg("mid9DALnQhYqcFxfmUjPm1HX", "error.log")
path = os.path.join(os.getcwd(),'images')#图片放到程序的同级文件夹images 里面
for pic in os.listdir(path):
	#print(pic)
    rmbg.remove_background_from_img_file(f"{path}\{pic}")