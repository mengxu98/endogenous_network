# -*- coding: utf-8 -*-
'''
Created on Mon Sep 26 11:45:11 2016

Counting for all attracotors without two nodes.

@author: jiang
'''
import network_frame
import itertools as it
import sys,os
from random import *

def main(Input_lst = []):   
    folder = int(sys.argv[1])
    total_num = int(sys.argv[2])+1
    attracotors_l = []
    if sys.argv[1] != str(0):
        num = list(it.combinations(range(1,total_num),folder))
        for i in num:
            filename = [str(i[j]) for j in range(folder)]
            network_frame.Attractor_Calculation('network_without_' + '_'.join(filename) + '.xlsx',
                                                str(folder),
                                                'network_without_' + '_'.join(filename) + '_attractors',
                                                Input_lst)
    else:
        network_frame.Attractor_Calculation('network_without_0.xlsx', str(folder), 'network_without_0_attractors',Input_lst)
            

# create input_lst from list
try:
    inputfile = sys.argv[3]
    f = open(os.path.join(os.getcwd(),inputfile),'r')
    Input_lst = [random() for i in range(42)]
    for i in f.readlines():
        index = int(i.split('\t')[0])-1
        Input_lst[index] = float(i.split('\t')[1])
    main(Input_lst)
except:
    main([])
