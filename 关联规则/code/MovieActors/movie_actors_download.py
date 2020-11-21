# -*- coding: utf-8 -*-
# 下载某个演员/导演的电影数据集
from lxml import etree
import time
from selenium import webdriver
import pandas as pd

"""
这里我们需要使用ChromeDrvier来做模拟
Step1，打开谷歌浏览器， 在地址栏输入 chrome://version/  查看版本信息
Step2，ChromeDriver版本下载地址：http://chromedriver.storage.googleapis.com/index.html
Step3，放到Python\Lib\site-packages相应路径
"""
#chrome_driver = r"C:\Python37\Lib\site-packages\selenium\webdriver\chrome\chromedriver.exe"
#driver = webdriver.Chrome(executable_path=chrome_driver)
driver = webdriver.Chrome()
# 设置想要下载的导演 数据集
director = u'徐峥'
base_url = 'https://movie.douban.com/subject_search?search_text='+director+'&cat=1002&start='

movie_actors = {}
# 下载指定页面的数据
def download(request_url):
	driver.get(request_url)
	time.sleep(1)
	html = driver.find_element_by_xpath("//*").get_attribute("outerHTML")
	html = etree.HTML(html)
	# 设置电影名称，导演演员 的XPATH
	movie_lists = html.xpath("/html/body/div[@id='wrapper']/div[@id='root']/div[1]//div[@class='item-root']/div[@class='detail']/div[@class='title']/a[@class='title-text']")
	name_lists = html.xpath("/html/body/div[@id='wrapper']/div[@id='root']/div[1]//div[@class='item-root']/div[@class='detail']/div[@class='meta abstract_2']")
	# 获取返回的数据个数
	num = len(movie_lists)
	
	if num > 15: #第一页会有16条数据
		# 默认第一个不是，所以需要去掉
		movie_lists = movie_lists[1:]
		name_lists = name_lists[1:]
	for (movie, name_list) in zip(movie_lists, name_lists):
		# 会存在数据为空的情况
		if name_list.text is None: 
			continue
		# 显示下演员名称
		names = name_list.text.split('/')
		movie_actors[movie.text] = name_list.text.replace(" ", "")
		print(name_list.text.replace(" ", ""))
	print('OK') # 代表这页数据下载成功
	if num >= 15:
		# 继续下一页
		return True
	else:
		# 没有下一页
		return False

# 开始的ID为0，每页增加15
start = 0
while start<10000: #最多抽取1万部电影
	request_url = base_url + str(start)
	# 下载数据，并返回是否有下一页
	flag = download(request_url)
	if flag:
		start = start + 15
	else:
		break

# 将字典类型转化为DataFrame
movie_actors = pd.DataFrame(movie_actors, index=[0])
#print(movie_actors)
# DataFrame 行列转换
movie_actors = pd.DataFrame(movie_actors.values.T, index=movie_actors.columns, columns=movie_actors.index)
movie_actors.index.name = 'title'
movie_actors.set_axis(['actors'], axis='columns', inplace=True)
movie_actors.to_csv('./movie_actors.csv')
print('finished')
