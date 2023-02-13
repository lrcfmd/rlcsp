Requirements:

1) Python3.6 or later

once you have python3 installed, you need to download the following 
dependencies all of which can be installed via pip:

1) numpy
2) math
3) statistics
4) matplotlib
5) time
6) random
7) mysql
8) mysql-connector-python-rf
9) scipy
10) heapq

You can then install RLCSP by typing:

python3 setup.py install

################################################################################

In order to run RLCSP, you will require to have access to a MySQL server.
You can install a MySQL server on local machine using this guide:
https://dev.mysql.com/doc/mysql-getting-started/en/#mysql-getting-started-installing

################################################################################

To use RLCSP in FUSE or MC-EMMA you need first to download and install FUSE (https://github.com/lrcfmd/FUSE_RL.git)
or MC-EMMA (https://github.com/lrcfmd/MC-EMMA-RL). Then fill fields 'host', 'database', 'user', 'password' 
to access your MySQL Server in the chosen input file. Then you can run the input file.

