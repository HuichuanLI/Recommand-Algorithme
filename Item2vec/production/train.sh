train_file=$1
item_vec_file=$2
../bin/word2vec -train $train_file -output $item_vec_file -size 128 -window 5 -sample 1e-3 -negative 5 -hs 0 -binary 0 -cbow 0 -iter 50
