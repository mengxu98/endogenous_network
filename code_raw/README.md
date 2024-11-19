Read Me
--------------------------
Prepare:

1.Put following files into one directory:
  'network_rewiring.py',
  'network_frame.py',
  'attractors_without_nodes.py'，
  'network_originon.xlsx'

2.Make sure following python packages have been installed on your device:
  scipy,
  numpy,
  pandas,
  itertools
  
Steps:

1.Run script 'network_rewiring.py' and add the number of nodes you want to delete,for example:
      'python network_rewiring.py 1',          
  which means you should run the script using commond line.
  This step will create the folder and files needed in the following steps.
  
2.
  1)Without start input
  Run script 'attractors_without_nodes.py' and add the number of nodes you want to delete,the total number of network 
  for example:
      'python attractors_without_nodes.py 1 42',
  '1' means you want to delete 1 node, '42' means the origion network contains 42 nodes.Also, commond line is needed.
  This step will calculate the attractors and put them into the folder named as the number of nodes you input,for example:
      '/../network_without_nodes/2'
  The attractors are something like follows:
      691	['0.0000', '0.0000', '1.0000', '0.0000', '0.0000', '0.9543', '0.9306', '0.9306', '0.0000', '0.8402', '0.0000', '0.8259', '0.8184', '0.9170', '0.8694', '0.9405', '0.0000', '0.0000', '0.0000', '0.0000', '0.0000', '0.0000', '0.0000', '0.0000', '0.0000', '0.0000', '1.0000', '0.0000', '1.0000', '0.0000', '0.0000', '0.0000', '0.0000', '0.0000', '0.0000', '0.0000', '0.0000', '0.0000', '0.0000', '0.0000', '0.0000', '0.0000']
      128	['0.0000', '0.0000', '1.0000', '0.0000', '0.0000', '0.9543', '0.9306', '0.9306', '0.0000', '0.0000', '0.0000', '0.0000', '0.0000', '0.0000', '0.0000', '0.0000', '0.0000', '0.0000', '0.0000', '0.0000', '0.0000', '0.0000', '0.0000', '0.0000', '0.0000', '0.0000', '1.0000', '0.0000', '1.0000', '0.0000', '0.0000', '0.0000', '0.0000', '0.0000', '0.0000', '0.0000', '0.0000', '0.0000', '0.0000', '0.0000', '0.0000', '0.0000']   
  first column is the size of attractor, second column is the attractor.
  
  2)With start input
  Make sure inputfile exist in your working directory.
  Run script 'attractors_without_nodes.py' and add the number of nodes you want to delete,the total number of network and the inputfilename.
  for example:
        'python attractors_without_nodes.py 1 42 Input_file.txt'
  '1' means you want to delete 1 node, '42' means the origion network contains 42 nodes,'Input_file.txt' is the name of your input.
  The inputfile should be like the following format:
        1	0.0
        2	0.0
        3	0.0
        4	0.0
        5	0.0
        6	0.0
        ...
        33	0.0
        34	0.0
        35	0.0
        36	0.0
        37	0.0
        38	0.0
        39	0.0
        40	0.0
        41	0.0
        42	0.0
   left is the nodes number, it should start from 1;
   right is its value and it should be float, using space to sepreate them.
   Others are the same like 2.1.