"""
对电影数据进行演员信息的扩展
"""

import os

import requests
from tqdm import tqdm


def _get_movie_author(title):
    url = "https://v2.sg.media-imdb.com/suggestion"
    if '(' in title:
        # 括号之后的内容全部不要
        title = title[:title.index('(')].strip()
    name = title.replace(' ', '_').replace(",", "").replace("?", "")
    url = f"{url}/{name[0].lower()}/{name[:18]}.json"
    response = requests.get(url)
    if response.status_code == 200:
        # 直接将第一个视频里面的演员作为匹配的演员
        data = response.json()
        authors = data['d'][0]['s']
        authors = authors.split(",")
        authors = [author.strip() for author in authors]
        authors = ','.join(authors)
        return authors
    else:
        return ""


def append_authors(root_dir, name, new_name):
    path = os.path.join(root_dir, name)
    path2 = os.path.join(root_dir, new_name)
    with open(path, 'r', encoding='utf-8') as reader:
        with open(path2, 'w', encoding='utf-8') as writer:
            for _ in tqdm(range(1682)):
                line = reader.readline().strip()
                if line is None:
                    break
                arrs = line.split("|", maxsplit=24)
                try:
                    author = _get_movie_author(arrs[1])
                except:
                    print(arrs[1])
                    author = ""
                arrs.append(author)
                line = '\t'.join(arrs)
                writer.writelines(f"{line}\n")
