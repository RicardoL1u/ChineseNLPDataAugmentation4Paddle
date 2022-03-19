#!/usr/bin/env python
#encoding=utf-8
'''
@Time    :   2022/03/19 23:31:30
@Author  :   RicardoL1u
@Contact :   ricardoliu@outlook.com
'''

# here put the import lib

def read_file(file_p):
    out_arr = []
    with open(file_p, encoding='utf-8') as f:
        out_arr = [x.strip() for x in f.readlines()]
    return out_arr

