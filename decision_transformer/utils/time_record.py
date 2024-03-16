#!/usr/bin/python
# -*- coding:utf8 -*-

import time
class TimeRecorder:
    def __init__(self):
        self.infos = {}

    def __call__(self, info, *args, **kwargs):
        class Context:
            def __init__(self, recoder, info):
                self.recoder = recoder
                self.begin_time = None
                self.info = info

            def __enter__(self):
                self.begin_time = time.time()

            def __exit__(self, exc_type, exc_val, exc_tb):
                self.recoder.infos[self.info] = time.time() - self.begin_time

        return Context(self, info)

    def __str__(self):
        return ' '.join(['{}:{:.2f}s'.format(info, t) for info, t in self.infos.items()])

    def __getitem__(self, item):
        return self.infos[item]


if __name__ == '__main__':
    a = 112312341241
    b = 12341235412
    tr = TimeRecorder()

    with tr('add'):
        for i in range(1000000):
            x = a + b
    with tr('multiply'):
        for i in range(1000000):
            y = a * b

    print(tr)