#!/usr/bin/env python
# -*- coding: utf-8 -*-

# 基础模块
import os
import sys
import time

ts = time.time()

num = sys.argv[1]

sub_file_path = '../../../output/m3/lgb_m3_{}'.format(num)
sub_name = 'lgb_m3_{}'.format(num)

# lgb train
print ('lgb_train-%s.py %s' % (num, num))
os.system('python3 -u lgb_train_%s.py %s' % (num, num))

# merge cv & sub
print('\nkfold_merge')
os.system('python3 -u kfold_merge.py %s %s' % (sub_file_path, sub_name))

# convert cv & sub to list format
print ('\nconvert')
os.system('python3 -u convert.py %s %s' % (sub_file_path, sub_name))

# calculate mrr & auc
print ('\neval')
os.system('python3 -u eval.py %s' % ('{}/r_{}_cv.csv'.format(sub_file_path, sub_name)))

print ('all completed, cost {}s'.format(time.time() - ts))
