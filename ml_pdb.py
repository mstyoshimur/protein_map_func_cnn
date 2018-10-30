#!/usr/bin/python3

import os 
from time import sleep,ctime
import gzip
from random import random,randint,shuffle
from signal import signal,SIGINT

import numpy as np
import tensorflow as tf
import subprocess
import struct
import shutil
import heapq

import keras
#from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Activation
from keras.layers import Conv2D, Conv3D, MaxPooling2D, MaxPooling3D
from keras.layers.advanced_activations import LeakyReLU
from keras import backend as K


N=128

def getFileList(path):
  file_list=[]
  for (root,dirs,files) in os.walk(path):
   for file in files:
     file_list.append(os.path.join(root,file))
  return file_list

def pdbtohkl(pdbf,hkltemp,i):
  cmd1 = "/work/ccp4/ccp4-7.0/bin/sfall xyzin %s  hklout  %s  << EOF >/dev/null \n"%(pdbf,hkltemp)  
  cmd1_2 = "MODE SFCALC XYZIN\n"+"SYMM P212121\n"+"RESOLUTION 10 4.5 \n"+"END\n" +"EOF\n"
  return subprocess.call(cmd1+cmd1_2,shell=True)
  
def hkltomap(hkltemp,mapout,i,cells):
#  hkltemp = "/ramdisk/hkl%d.mtz"%i
  maptemp1 = "/ramdisk/map1_%d.map"%i
  cmd2 = "cfft -mtzin %s -mapout %s -colin-fc FC,PHIC > /dev/null \n"%(hkltemp,maptemp1)
  subprocess.call(cmd2,shell=True)
  if os.path.exists(maptemp1):
       cmd3 = "mapmask mapin %s mapout %s  << EOF >/dev/null \n"%(maptemp1,mapout)
       cmd3_2 ="XYZLIM CELL \n"+"END\n" + "EOF\n"
       subprocess.call(cmd3+cmd3_2,shell=True)
       if os.path.exists(mapout):
          os.remove(hkltemp)
          os.remove(maptemp1)  

def hkltoranmap(hkltemp,mapout,i,cells,degran):

  if abs(degran) >  0.001: 
    maptemp1 = "/ramdisk/map1_%d.map"%i
    hkltemp2="/ramdisk/hklran1_%d.mtz"%i
    currenthkl="/ramdisk/curhklran.mtz"
    currentmap="/ramdisk/curran.map"
    
    if os.path.exists(hkltemp2):
       os.remove(hkltemp2)

    cmd1a   = 'sftools\n'
    cmd1a_=['' for i in range(20)]
    p=subprocess.Popen(cmd1a.encode("UTF-8"),stdin=subprocess.PIPE,stdout=subprocess.DEVNULL,shell=True)

    seed=randint(0,9999)
    cmd1a_[0] = 'CALC SEED %d \n'%seed
    cmd1a_[1] = 'READ %s \n'%hkltemp
    cmd1a_[2] = 'SELECT not centro \n'
    cmd1a_[3] = 'CALC P COL PHICR = ran_u 6.2831853 *  \n'
    cmd1a_[4] = 'SELECT ALL  \n'
    cmd1a_[5] = 'SELECT centro  \n'
    cmd1a_[6] = 'CALC P COL PHICR = COL PHIC   \n'
    cmd1a_[7] = 'SELECT ALL  \n'
    cmd1a_[8] = 'WRITE %s COLUMN FC PHICR \n'%hkltemp2
    cmd1a_[9] = 'STOP\n'

    for icmd1 in range(10):
      p.stdin.write(cmd1a_[icmd1].encode("UTF-8"))
    p.stdin.close()
    p.wait()
    cmd2 = "cfft -mtzin %s -mapout %s -colin-fc FC,PHICR > /dev/null \n"%(hkltemp2,maptemp1)
    
  else: # fabs(degran) == 0
    maptemp1 = "/ramdisk/map0_%d.map"%i
    hkltemp2="/ramdisk/hklran0_%d.mtz"%i
    currenthkl="/ramdisk/curhkl.mtz"    
    currentmap="/ramdisk/curent.map"
    
    shutil.copyfile(hkltemp,hkltemp2)
    cmd2 = "cfft -mtzin %s -mapout %s -colin-fc FC,PHIC > /dev/null \n"%(hkltemp2,maptemp1)
    
  if  os.path.exists(hkltemp2):
    
      subprocess.call(cmd2,shell=True)
      if os.path.exists(maptemp1):
         cmd3 = "mapmask mapin %s mapout %s  << EOF >/dev/null \n"%(maptemp1,mapout)
         cmd3_2 ="XYZLIM CELL \n"+"END\n" + "EOF\n"
         subprocess.call(cmd3+cmd3_2,shell=True)
         if os.path.exists(mapout):
              shutil.copyfile(maptemp1,currentmap)           
              os.remove(maptemp1)
         if os.path.exists(hkltemp2):
              shutil.copyfile(hkltemp2,currenthkl)           
              os.remove(hkltemp2)
         return 0
  else:
         return 1          
     
print(K.image_data_format())          
input_shape = (40,40,40,1)
num_classes = 2
epochs = 16
leaky_relu = LeakyReLU(0.1)
#relu = ReLU(0.1)
model = Sequential()
model.add(Conv3D(32, kernel_size=(5,5,5), strides=(2,2,2), activation='relu',input_shape=input_shape))  # Layer1 conv3D 1
#model.add(LeakyReLU(0.1)) #Layer2 Activate
model.add(MaxPooling3D(pool_size=(2,2,2))) #Layer
model.add(Dropout(0.1)) # Layer? dropout
#model.add(Conv3D(128, kernel_size=(5,5,5), strides=(1,1,1), activation='relu')) # Layer3 conv3D 2
model.add(Conv3D(64, kernel_size=(3,3,3), strides=(1,1,1), activation='relu'))# Layer3 conv3D 2
#model.add(LeakyReLU(0.1))#Layer 4 Activate
model.add(MaxPooling3D(pool_size=(2,2,2))) #Layer 5 maxpool
model.add(Dropout(0.1))#Leyer? dropout
#model.add(Conv3D(256, kernel_size=(3,3,3), strides=(1,1,1),activation='relu')) # Layer3 conv3D 2
#model.add(Conv3D(256, kernel_size=(3,3,3), strides=(1,1,1),activation='relu')) # Layer3 conv3D 2
#model.add(Conv3D(256, kernel_size=(3,3,3), strides=(1,1,1),activation='relu')) # Layer3 conv3D 2
#model.add(LeakyReLU(0.1))#Layer 4 Activate
#model.add(MaxPooling3D(pool_size=(2,2,2))) #Layer 5 maxpool
#model.add(Conv3D(256, kernel_size=(3,3,3), strides=(1,1,1),activation='relu')) # Layer3 conv3D 2
#model.add(Conv3D(256, kernel_size=(3,3,3), strides=(1,1,1),activation='relu')) # Layer3 conv3D 2
#model.add(Conv3D(256, kernel_size=(3,3,3), strides=(1,1,1),activation='relu')) # Layer3 conv3D 2
#model.add(LeakyReLU(0.1))#Layer 4 Activate
#model.add(MaxPooling3D(pool_size=(2,2,2))) #Layer 5 maxpool
#model.add(Dropout(0.4))#Leyer? dropout
model.add(Flatten())#Layer6 Flattern
model.add(Dense(128,activation='linear')) #Layer7 Dense
model.add(Dropout(0.3))#Layer? dropout
model.add(Dense(num_classes, activation='linear')) #Layer8 Dense
model.add(Activation("softmax"))          
i=0
j=0
k=0
cur_result2 = [0.,0.]
x_train=np.ndarray(shape=(N,40,40,40,1))
y_train=np.ones(shape=(N,2))
x_train = x_train.astype('float32')
y_train = y_train.astype('int32')
#model.compile(loss=keras.losses.categorical_crossentropy,optimizer=keras.optimizers.Adam(),metrics=['accuracy'])
model.compile(loss=keras.losses.categorical_crossentropy,optimizer=keras.optimizers.SGD(lr=0.001,momentum=0.9,decay=0.00016666,nesterov=False),metrics=['accuracy'])
np.set_printoptions(precision=4)
filelist=getFileList("./")
#shuffle(filelist)
for ifile, file in enumerate(filelist):
#   print( "\n",file)
 XRAY=0
# if 32001<ifile < 32810:
 if ifile < 0:
   pass
 else:
   if i > 2000000:
      os.remove("/work/mlpdb/run")
   if not os.path.exists("/work/mlpdb/run"):
      break
   if file[-3:] == ".gz":
#       print( " select ")
    with gzip.open(file,'rb') as f:
     SPAC=0
     for line in f.readlines():
         words=line.split()
         if words[0] == b"EXPDTA":
           if b'X' in  words[1] :
              XRAY=1
         if words[0]==b"CRYST1":
           strspac=line[55:65]
           if strspac == b"P 21 21 21":
              SPAC=19
              cella = float(words[1])
              cellb = float(words[2])
              cellc = float(words[3])
              cells = [cella,cellb,cellc]
           break
     
     if SPAC == 19 and 10. < min(cells) and max(cells) < 200.:

        pdbf='/ramdisk/temp%d.pdb'%ifile
        hklf='/ramdisk/temp0_%d.mtz'%ifile

        print( " XRAY ",strspac,i,ifile,file)
        in_f = gzip.GzipFile(file,'rb')
        out_f = open(pdbf,'w')
        out_f.write(str(in_f.read())[2:-1].replace("\\n","\n"))
        out_f.close()
        in_f.close()


#        pdbtomap(pdbf,mapif,i,cells)
#        print('current file is '+file)
        result=pdbtohkl(pdbf,hklf,ifile)
        if result != 0:
#          shutil.copyfile('/ramdisk/sfalllog','/ramdisk/sfallerr')
#          shutil.copyfile(pdbf,'/ramdisk/sfallerrpdb.pdb')
           pass
        if result == 0:
          i=i+1
          ab_pdb = False
          for iran in [0,1]:
            if ab_pdb == True:
              break
            if iran == 1: 
               dran=10.
            else:
               dran=0.
            mapif='/ramdisk/temp%d_%d.map'%(i,iran)
            hkltoranmap(hklf,mapif,ifile,cells,dran)


            if os.path.exists(mapif):        
              mapb=open(mapif,'rb')
              NC,NR,NS=struct.unpack_from("III",mapb.read(),0)
  
              mapb.seek(1344,0) # 1344 = 1024+320 header + symmP212121
              mapmat=np.fromfile(mapb,np.float32,-1).reshape((NS,NR,NC,1))
  
              iNC,iNR,iNS = int(40/NC)+1, int(40/NR)+1, int(40/NS)+1
  
              mapext=np.tile(mapmat,(iNS,iNR,iNC,1))

              mapnrm=mapext[:40,:40,:40,:1]

#              depth=mapnrm.max()-mapnrm.min()
#              mintemp=mapnrm.min()
#              maxv =mapnrm.max()
#              minv =mapnrm.min()
#              depth=maxv-minv
              sigma=np.std(mapnrm)
              ave=np.mean(mapnrm)
              if sigma <= 0.01:
                shutil.copyfile(mapif,'/ramdisk/error.map')
                shutil.copyfile(pdbf,'/ramdisk/error.pdb')
                print(ifile,file,'skipped\n')
                ab_pdb=True
                break
              mapnrmnrm = (mapnrm-ave)*0.2/sigma 
              np.clip(mapnrmnrm,-1.,1.,out=mapnrmnrm)              
#              print(mapnrm[0][0],mapnrmnrm[0][0])
              x_train[j]=mapnrmnrm
              y_train[j]=[1-iran,iran] #[1,0] correct phase, [0,1] random phase


              j=j+1

              if os.path.exists(mapif):
                 os.remove(mapif)
              if j >= N :
                if  k % 10 == 1:
                   cur_result2=model.evaluate(x_train,y_train)
                   for jj in range(N):
                     
#                     print("ave %8.3f "%x_train[jj].mean(),end="")
#                     print("std %8.3f  "%x_train[jj].std(),end="")
                     print(y_train[jj],end="")
                     print("  ",end="")
                     pripre=model.predict(x_train)[jj]

                     print(pripre,end="")
                     if y_train[jj][0] == 1 and y_train[jj][1] == 0:
                        print(" correct ")
                     else:
                        print(" random  ")

                   print("loss= %8.3f "%cur_result2[0],"precise = %8.3f "%cur_result2[1])
                   logtrainf=open('/work/mlpdb/prog/log_train','a')
#                   logtrainf.write('i= %d '%i+'loss= %f '%cur_result2[0]+' precise= %f \n'%cur_result2[1])
                   logtrainf.write('i= %d '%i+'loss= %f '%cur_result2[0]+'prec= %f \n'%cur_result2[1])
                   logtrainf.close()
#                   loss=cur_result2[0]
#                   print("model value %8.3f "%y_train[0][0],"predicted ",cur_result2[0][0])
                print('train_batch ',N,i)
                j = 0
                if -1 < cur_result2[0] < 20:
#                   print(y_train)
                   model.train_on_batch(x_train,y_train)

                   k = k+1
                else:
                   print("skip train \n ")
                
              
#             model.fit(x_train,y_train,batch_size=i,epochs=epochs,verbose=1)
                model.save('/work/mlpdb/save/p212121train')
                json_string = model.to_json()
                open('/work/mlpdb/save/p212121train.json','w').write(json_string)
                model.save_weights('/work/mlpdb/save/p212121train.h5')
#                model.summary()
#          print(i,dran)
        if os.path.exists(hklf):
             os.remove(hklf)
        if os.path.exists(pdbf):
          os.remove(pdbf)
   if XRAY != 1 :
          os.remove(file)
