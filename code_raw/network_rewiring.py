# -*- coding: utf-8 -*-
"""
Created on Mon Sep 26 09:40:18 2016

@author: jiang
"""
#
import pandas as pd
import sys,os

frame = pd.read_excel(os.path.join(os.getcwd(), 'network_originon.xlsx'),sheetname='Sheet1')

def network_rewiring(num,frame=frame):
    frame = frame.fillna('nothing')
    frame = frame.drop(num)
    for i in frame.index:#1,2...42
        for line in ['Activated by', 'Inhibited by']:
            if str(frame.loc[i,line]) != 'nothing':
                if ',' in str(frame.loc[i,line]):
                    lst = str(frame.loc[i,line]).split(',')

                    if str(num) in lst:
                        lst.remove(str(num))
                    if len(lst) != 0:
                        frame.loc[i,line] = ','.join(lst)

                else:   
                    if str(frame.loc[i,line]) == str(num):
                        frame.loc[i,line] = 'nothing'
    return frame
    
def build(i):
    # print i
    frame_temp =  network_rewiring(i[0])
    l = len(i) #ex,l = 
    for time in range(1,l):
        try:
            frame_temp =  network_rewiring(i[time], frame_temp)
        except:
            pass
        for ind in frame_temp.index:
            if (frame_temp.loc[ind][0] == 'nothing') and (frame_temp.loc[ind][1] == 'nothing'):
                try:
                    frame_temp = network_rewiring(ind,frame_temp)
                except:
                    pass
    filename = [str(i[j]) for j in range(l)]
    frame_temp.to_excel(os.path.join(os.getcwd(),
                        str(gnodesnum),
                        'network_without_' + '_'.join(filename) + '.xlsx'))
                        # 'network_without_' + str(i[0]) + '_' + str(i[1]) + '.xlsx'))
                        
def main(nodesnum, totalnum): # Number of nodes being excluded, Number of total nodes
    import itertools as it
    num = list(it.combinations(range(1,totalnum),nodesnum))
    # print num
    #num = [(1,9)]
    try:
        os.makedirs(os.path.join(os.getcwd(),str(nodesnum)))
    except:
        pass
    gnodesnum = nodesnum
    
    global gnodesnum

    import time
    time1 = time.time()
    from multiprocessing import Pool as ThreadPool
    pool = ThreadPool()  
    pool.map(build,num)
    # for i in num:
        # build(i)
    time2 = time.time()
    print time2 - time1
    
if __name__ == '__main__': #直接运行.py文件
    # print sys.argv[1],type(sys.argv[1])
    folder = '0'
    folder_path = os.path.join(os.getcwd(), folder)
    try:
        os.makedirs(folder_path)
    except:
        pass
    if sys.argv[1] == str(0):
        print 'Origion network is running'
        frame = frame.fillna('nothing')
        frame.to_excel(os.path.join(os.getcwd(),
                       str(0),
                       'network_without_' + str(0) + '.xlsx'))
    else:
        main(int(sys.argv[1]), len(frame.index)+1)