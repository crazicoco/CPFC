#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Organization  : Southeast University
# @Author        : 作者
# @Time          : 2020/12/27 22:54
# @Function      : valid the label predicted from model if is same . is same
import json


def main():
    dataset = ['ocemotion', 'ocnli', 'tnews']
    root = "/share/home/crazicoco/competition/CPFC/record_analysis"
    for name in dataset:
        flag = 0
        wrong_ids = []
        with open(root + '/2020_12_28_14_48/submission/'+name + '_predict.json', "r") as f:
            with open(root + '/2020_12_26_0_14/submission/'+name + '_predict.json', "r") as f_:
                    lines = f.readlines()
                    lines_ = f_.readlines()
                    for idx in range(len(lines)):
                        line = json.loads(lines[idx])
                        line_ = json.loads(lines_[idx])
                        if line['label'] != line_['label']:
                            flag += 1
                            wrong_ids.append(int(line['id']))
                    if not flag:
                        print("there no different sample predict")
                    else:

                        print(flag)
                        print(wrong_ids)


if __name__ == '__main__':
    main()