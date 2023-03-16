This has been tested on unix based systems.

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
8) mysql-connector-python-rf (this will install mysql in python as well)
9) scipy
10) heapq

You can then install RLCSP by typing:

python3 setup.py install

################################################################################

In order to run RLCSP, you will require to have access to a MySQL server.
You can install a MySQL server on local machine using this guide:
https://dev.mysql.com/doc/mysql-getting-started/en/#mysql-getting-started-installing
NOTE: you will need to set a password for the root user when installing mysql.

################################################################################

To run on a local machine, once mysql is installed as above you will need to set up a 
local user for your install of mysql, using the root user accound created when mysql
was install. From a command terminal type the following:

sudo mysql -u root -p

You will need to enter the sudo password for your machine and then the roos password
setup when mysql was installed. Then enter the following

CREATE USER 'myuser'@'localhost' IDENTIFIED WITH mysql_native_password BY 'mypassword';

Where ‘myuser’ should be replaced with the username that you want to use run the rl-csp 
codes, and ‘mypassword’ is your chosen password for the database.


################################################################################

To use RLCSP in FUSE or MC-EMMA you need first to download and install FUSE (https://github.com/lrcfmd/FUSE_RL)
or MC-EMMA (https://github.com/lrcfmd/MC-EMMA-RL). Then fill fields 'host', 'database', 'user', 'password' 
to access your MySQL Server in the chosen input file. Then you can run the input file. 

################################################################################

The input files for each policy-composition pair tested in the paper "Reinforcement Learning in Cystal Structure Prediction" are located in the folders 'input_FUSE' and 'input_MCEMMA' and can be used to reproduce the results of the paper. The inpute files are named as 'input_<CODE>_<policy>_<composition>.py' where <CODE> is either FUSE or MCEMMA, <policy> is one of three policies (RLCSP, Original, Uniform), and <composition> is one of six compositions (Sr4Ti3O10, Y2TiO5, Y2Ti2O7, Sr2YO4, Y2O3, YBa2Ca2Fe5O13). 
To run FUSE with the Original or Uniform policies you need to install the corresponding version of FUSE; TODO:LINK for the Original policy or TODO:LINK for the Uniform policy.
To run MC-EMMA with the Original or Uniform policies you need to install MC-EMMA from https://github.com/lrcfmd/MC-EMMA-stable .

################################################################################

Removing mysql databases that you are finished with

If you wish to remove databases from your machine, you can login to mysql from the command line:

mysql -u 'myuser' -p

and then use the following command to remove a data base as set with "database" above:

DROP DATABASE "database";


