#coding: utf-8
import cPickle as pickle
import sys

def parse_playlist_get_info(in_line, playlist_dic, song_dic):
	contents = in_line.strip().split("\t")
	name, tags, playlist_id, subscribed_count = contents[0].split("##")
	playlist_dic[playlist_id] = name
	for song in contents[1:]
		try:
			song_id, song_name, artist, popularity = song.split(":::")
			song_dic[song_id] = song_name
		except:
			print "song format error"
			print song+"\n"

		

def parse_file(in_file, out_file):
	#从歌单id到歌单名称的映射字典
	playlist_dic = {}
	#从歌曲id到歌曲名称的映射字典
	song_dic = {}
	for line in open(in_file):
		parse_playlist_get_info(line, playlist_dic, song_dic)
	#把映射字典保存在二进制文件中
	pickle.dump(playlist_dic, open("playlist.pkl","wb")) 
	#可以通过 playlist_dic = pickle.load(open("playlist.pkl","rb"))重新载入
	pickle.dump(song_dic, open("song.pkl","wb"))