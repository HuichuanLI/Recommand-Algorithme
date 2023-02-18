#coding: utf-8
import multiprocessing
import gensim
import sys

def parse_playlist_get_sequence(in_line, playlist_sequence):
	song_sequence = []
	contents = in_line.strip().split("\t")
	# 解析歌单序列
	for song in contents[1:]:
		try:
			song_id, song_name, artist, popularity = song.split(":::")
			song_sequence.append(song_id)
		except:
			print "song format error"
			print song+"\n"
	playlist_sequence.append(song_sequence)


def train_song2vec(in_file, out_file):
	#所有歌单序列
	playlist_sequence = []
	#遍历所有歌单
	for line in open(in_file):
		parse_playlist_get_sequence(line, playlist_sequence)
	#使用word2vec训练
	cores = multiprocessing.cpu_count()
	print "using all "+str(cores)+" cores"
	print "Training word2vec model..."
	model = gensim.models.Word2Vec(sentences=playlist_sequence, size=150, min_count=1, window=3, workers=cores)
	print "Saving model..."
	model.save(out_file)

def test_song2vec_model(model, song_id):


if __name__ == '__main__':
	in_file = sys.argv[1]
	out_file = sys.argv[2]
	train_song2vec(in_file, out_file)