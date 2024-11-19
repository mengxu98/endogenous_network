# -*- coding: utf-8 -*-
"""
Spyder Editor

parallel test

This is a temporary script file.
"""


import pandas as pd
import numpy as np
from scipy.integrate import odeint
import os
# import solve

## -------------------------- ##
# Funciton production can't be put into function,cause it will be produced repeatedly.
## -------------------------- ##
def function_producion(frame):
# inputfile = 'network_originon.xlsx'
# frame = pd.read_excel(os.path.join(os.getcwd(), inputfile),sheetname='Sheet1')
    # import numpy as np
    function = []
    v1 = ''
    v2 = ''
    ep = 'def solve(w,t,n):'+'\n'
    for row in range(len(frame.index)):
        p1 = unicode(frame.iat[row,0])
        p2 = unicode(frame.iat[row,1])
        if p1!='nothing':
            if ',' in p1:
                p1 = p1.split(',')
    #            p1 = ['x' + str(dic[item.strip()]) for item in p1]
                p1 = ['x' + str(item.strip()) for item in p1]
                p1 = '+'.join([item+'**3' for item in p1])
                p1 = '8*(' + p1 + ')/(1 + 8*(' + p1 + '))'
            else:
    #            p1 = 'x' + str(dic[p1.strip()])
                p1 = 'x' + str(p1.strip())
                p1 = '8*('+ p1 + '**3)/(1 + 8*(' + p1 + '**3))'
        else:
            p1 = ''
        if p2!='nothing':
            if ',' in p2:
                p2 = p2.split(',')
    #            p2 = ['x' + str(dic[item.strip()]) for item in p2]
                p2 = ['x' + str(item.strip()) for item in p2]
                p2 = '+'.join([item+'**3' for item in p2])
                p2 = '1/(1 + 8*(' + p2 + '))'
            else:
    #            p2 = 'x' + str(dic[p2.strip()])
                p2 = 'x' + str(p2.strip())            
                p2 = '1/(1 + 8*(' + p2 + '**3))'
        else:
            p2 = ''
    #    num = str(frame.index[row][1])
        num = str(frame.index[row])
        if p1!= '':
            if p2 != '':
                f = 'dx' + num + 'dt=' + p1 + '*' + p2 + '-x' + num
            else:
                f = 'dx' + num + 'dt=' + p1 + '-x' + num
        else:
            f = 'dx' + num + 'dt=' + p2 + '-x' + num
        v1 += 'x'+num+', '
        v2 += 'dx' + num + 'dt' +', '
        function.append('\t' + f + '\n')
    #    print f
    ep += '\t' + v1[:-2] + '= w' + '\n' + ''.join(function)
    ep += '\t' + 'return np.array(['+ v2[:-2] + '])' + '\n'
    # print ep 
    exec(ep)# in locals(), globals()
    solve2 = solve
    global solve2

def attractor(frame,start_list):
    import random
    t = np.arange(0, 100, 1)
    if start_list != []:
        start = tuple(start_list)
    else:
        start_range = len(frame[0].index)+1
        start = tuple([random.random() for i in range(1,start_range)]) 
    # print start,type(start),type(start[0])
    
    track1 = odeint(solve2, start, t, args=(4,))    
    # print track1
    # import matplotlib.pyplot as plt 
    # for i in range(len(track1[0])):
        # plt.plot(t,track1[:,i])
    # plt.show()
#    return track1[-1]
    return [('%.4f' % abs(item))for item in track1[-1]]

def network_frame(outputfile,frame,start_list):
    import time 
    time1 = time.time()
    from multiprocessing import Pool
    PPP = Pool()
    
    
    result = []
    for i in range(2000):
        r = PPP.apply_async(attractor,([frame],start_list))
        result.append(r)

    PPP.close()
    PPP.join()
    
    attractors = []
    attractors_dic = {}
    temp_lst = []
    
    for temp in result:
        temp = temp.get()
        if str(temp) not in temp_lst:
            attractors.append(temp)
            attractors_dic[str(temp)] = 0
            temp_lst.append(str(temp))
        else:
            attractors_dic[str(temp)] += 1

    # print attractors
    
    # attractors = []
    # temp_lst = []
    # for i in range(1000):
       # temp = attractor(9)
       # if str(temp) not in temp_lst:
           # attractors.append(temp)
           # temp_lst.append(str(temp))

    
    f = open(os.path.join(os.getcwd(),outputfile+'.txt'),'w+')
    for i in attractors_dic:
        f.write(str(attractors_dic[i]) + '\t' + str(i) + '\n')
    f.close()
    time2 = time.time()
    print time2 - time1
    return len(attractors)
    
def Attractor_Calculation(inputfile = 'network_without_0.xlsx',
                          folder = '0', 
                          outputfile='Attractors',start_list = []):
    # print start_list,'Attractor_Calculation is running well'
    # Create folder
    folder_path = os.path.join(os.getcwd(), folder)
    try:
        os.makedirs(folder_path)
    except:
        pass
       
    frame = pd.read_excel(os.path.join(os.getcwd(), folder, inputfile),
                          sheetname='Sheet1')
    function_producion(frame)
    outputfile = os.path.join(folder,outputfile)
    result = network_frame(outputfile,frame,start_list)
    
if __name__  == '__main__':
    Attractor_Calculation('network_without_0.xlsx','0','Attractors')