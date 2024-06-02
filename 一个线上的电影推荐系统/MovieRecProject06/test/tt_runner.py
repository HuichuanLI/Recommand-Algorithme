# -*- coding: utf-8 -*-
import os
import sys

sys.path.append(os.path.abspath("../src"))

if __name__ == '__main__':
    from movie_online.app import app

    app.run(host='0.0.0.0', port=5051)
