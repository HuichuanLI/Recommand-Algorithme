# -*- coding: utf-8 -*-
import sys
import os

sys.path.append(os.path.abspath("../src"))


def t0_user_cf():
    from movie_offline.models import user_cf
    # user_cf.download_data("../data/models/user_cf/training.txt")
    # NOTE: search_best_params 仅在模型成功上线之前会进行运行，用于选择最优模型超参数
    # user_cf.search_best_params('../data/models/user_cf/training.txt')
    user_cf.training("../data/models/user_cf/training.txt")
    # user_cf.timed_scheduling()


def t1_item_cf():
    from movie_offline.models import item_cf

    item_cf.training("../data/models/user_cf/training.txt")
    # item_cf.timed_scheduling()


def t2_mf():
    from movie_offline.models import mf

    mf.training("../data/models/user_cf/training.txt")
    # mf.timed_scheduling()


def t3_fm():
    from movie_offline.models.fm import fm_v2

    root_dir = '/Users/lhc456/Desktop/python/Recommand-Algorithme/Recommand-Algorithme/一个线上的电影推荐系统/MovieRecProject06/data/features'
    output_dir = '/Users/lhc456/Desktop/python/Recommand-Algorithme/Recommand-Algorithme/一个线上的电影推荐系统/MovieRecProject06/data/tmp/fm'

    # fm_v2.training(
    #     root_dir=root_dir,
    #     output_dir=output_dir
    # )

    # fm_v2.export(output_dir)

    # fm_v2.process_spu_embedding(
    #     root_dir=root_dir,
    #     model_dir=output_dir
    # )

    fm_v2.upload(
        model_dir=output_dir
    )

    fm_v2.deploy(
        model_dir=output_dir
    )


def t4_lr():
    from movie_offline.models.lr import lr

    root_dir = r'/Users/lhc456/Desktop/python/Recommand-Algorithme/Recommand-Algorithme/一个线上的电影推荐系统/MovieRecProject06/data/features'
    output_dir = r'/Users/lhc456/Desktop/python/Recommand-Algorithme/Recommand-Algorithme/一个线上的电影推荐系统/MovieRecProject06/data/tmp/lr'

    lr.training(
        root_dir, output_dir
    )
    lr.upload(
        model_dir=output_dir
    )


def t5_gbdt_lr():
    from movie_offline.models.gbdt_lr import gbdt_lr

    root_dir = r'/Users/lhc456/Desktop/python/Recommand-Algorithme/Recommand-Algorithme/一个线上的电影推荐系统/MovieRecProject06/data/features'
    output_dir = r'/Users/lhc456/Desktop/python/Recommand-Algorithme/Recommand-Algorithme/一个线上的电影推荐系统/MovieRecProject06/data/tmp/lr'

    gbdt_lr.training(
        root_dir, output_dir
    )
    gbdt_lr.upload(
        model_dir=output_dir
    )


def t6_bpr():
    from movie_offline.models.bpr import bpr
    root_dir = r'/Users/lhc456/Desktop/python/Recommand-Algorithme/Recommand-Algorithme/一个线上的电影推荐系统/MovieRecProject06/data/features'
    output_dir = r'/Users/lhc456/Desktop/python/Recommand-Algorithme/Recommand-Algorithme/一个线上的电影推荐系统/MovieRecProject06/data/tmp/lr'

    bpr.training(
        root_dir, output_dir
    )

    # bpr.export(
    #     model_dir=output_dir
    # )
    bpr.upload(
        model_dir=output_dir
    )


if __name__ == '__main__':
    t6_bpr()
