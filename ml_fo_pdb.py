#!/usr/bin/python3

import os 
from time import sleep,ctime
import gzip
from random import random,randint,shuffle,choice
from signal import signal,SIGINT

import numpy as np
import tensorflow as tf
import subprocess
import struct
import shutil
import heapq

import keras
#from keras.datasets import mnist
from keras.models import Sequential, load_model
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
  zerohalf=['0.0','0.5']
  tx=choice(zerohalf)
  ty=choice(zerohalf)
  tz=choice(zerohalf)
  cmd1_3 = "pdbset xyzin %s xyzout /ramdisk/tempshift.pdb << EOF>/dev/null\nSHIFT FRACTIONAL %s %s %s \nEND\nEOF\n"%(pdbf,tx,ty,tz)
  cmd1 = "/work/ccp4/ccp4-7.0/bin/sfall xyzin /ramdisk/tempshift.pdb  hklout  %s  << EOF >/ramdisk/sfalllog \n"%(hkltemp)  
  cmd1_2 = "MODE SFCALC XYZIN\n"+"SYMM P212121\n"+"RESOLUTION 10 4.5 \n"+"END\n" +"EOF\n"
  return subprocess.call(cmd1_3+cmd1+cmd1_2,shell=True)


def hkltomap(hkltemp,mapout,i,cells):
#  hkltemp = "/ramdisk/hkl%d.mtz"%i
  maptemp1 = "/ramdisk/mapfo1_%d.map"%i
  cmd2 = "cfft -mtzin %s -mapout %s -colin-fc FC,PHIC > /dev/null \n"%(hkltemp,maptemp1)
  subprocess.call(cmd2,shell=True)
  if os.path.exists(maptemp1):
       cmd3 = "mapmask mapin %s mapout %s  << EOF >/dev/null \n"%(maptemp1,mapout)
       cmd3_2 ="XYZLIM CELL \n"+"END\n" + "EOF\n"
       subprocess.call(cmd3+cmd3_2,shell=True)
       if os.path.exists(mapout):
          os.remove(hkltemp)
          os.remove(maptemp1)  

#def hkltoranmap(hkltemp,mapout,i,cells,degran):
def hklfotoranmap(hkltemp,hklfo,mapout,i,cells,degran):
  
    if abs(degran) >  0.001:
       ranstr="6.2831853"
    else:
       ranstr="0.0"
    maptemp1 = "/ramdisk/mapfo1_%d.map"%i
    hkltemp2="/ramdisk/hklforan1_%d.mtz"%i
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
    cmd1a_[3] = 'CALC P COL PHICR = ran_u %s *  \n'%ranstr
    cmd1a_[4] = 'SELECT ALL  \n'
    cmd1a_[5] = 'SELECT centro  \n'
    cmd1a_[6] = 'CALC P COL PHICR = COL PHIC   \n'
    cmd1a_[7] = 'SELECT ALL  \n'
    cmd1a_[8] = 'READ %s \n'%hklfo
    cmd1a_[9] = 'SELECT RESOL > 4.5 < 10.0 \nSELECT COL FP > 0 \n'
    cmd1a_[10] = 'WRITE %s COLUMN FP PHICR \n'%hkltemp2
    cmd1a_[11] = 'STOP\n'

    for icmd1 in range(12):
      p.stdin.write(cmd1a_[icmd1].encode("UTF-8"))
    p.stdin.close()
    p.wait()

    
    if  os.path.exists(hkltemp2):
      cmd2 = "cfft -mtzin %s -mapout %s -colin-fc FP,PHICR > /dev/null \n"%(hkltemp2,maptemp1)    
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
         if os.path.exists(hkltemp2):
              shutil.copyfile(hkltemp2,currenthkl)           
              os.remove(hkltemp2)  
         return 1          
     
#print(K.image_data_format())          
#input_shape = (40,40,40,1)
#num_classes = 2
#epochs = 16
#leaky_relu = LeakyReLU(0.1)
##relu = ReLU(0.1)
model = Sequential()
#model.add(Conv3D(32, kernel_size=(5,5,5), strides=(2,2,2), activation='relu',in#put_shape=input_shape))  # Layer1 conv3D 1
##model.add(LeakyReLU(0.1)) #Layer2 Activate
#model.add(MaxPooling3D(pool_size=(2,2,2))) #Layer
#model.add(Dropout(0.1)) # Layer? dropout
##model.add(Conv3D(128, kernel_size=(5,5,5), strides=(1,1,1), activation='relu')#) # Layer3 conv3D 2
#model.add(Conv3D(64, kernel_size=(3,3,3), strides=(1,1,1), activation='relu'))## Layer3 conv3D 2
##model.add(LeakyReLU(0.1))#Layer 4 Activate
#model.add(MaxPooling3D(pool_size=(2,2,2))) #Layer 5 maxpool
#model.add(Dropout(0.1))#Leyer? dropout
##model.add(Conv3D(256, kernel_size=(3,3,3), strides=(1,1,1),activation='relu'))# # Layer3 conv3D 2
##model.add(Conv3D(256, kernel_size=(3,3,3), strides=(1,1,1),activation='relu'))# # Layer3 conv3D 2
##model.add(Conv3D(256, kernel_size=(3,3,3), strides=(1,1,1),activation='relu'))# # Layer3 conv3D 2
##model.add(LeakyReLU(0.1))#Layer 4 Activate
##model.add(MaxPooling3D(pool_size=(2,2,2))) #Layer 5 maxpool
##model.add(Conv3D(256, kernel_size=(3,3,3), strides=(1,1,1),activation='relu'))# # Layer3 conv3D 2
##model.add(Conv3D(256, kernel_size=(3,3,3), strides=(1,1,1),activation='relu'))3 # Layer3 conv3D 2
##model.add(Conv3D(256, kernel_size=(3,3,3), strides=(1,1,1),activation='relu'))# # Layer3 conv3D 2
##model.add(LeakyReLU(0.1))#Layer 4 Activate
##model.add(MaxPooling3D(pool_size=(2,2,2))) #Layer 5 maxpool
##model.add(Dropout(0.4))#Leyer? dropout
#model.add(Flatten())#Layer6 Flattern
#model.add(Dense(128,activation='linear')) #Layer7 Dense
#model.add(Dropout(0.3))#Layer? dropout
#model.add(Dense(num_classes, activation='linear')) #Layer8 Dense
#model.add(Activation("softmax"))          
i=0
j=0
k=0
cur_result2 = [0.,0.]
x_train=np.ndarray(shape=(N,40,40,40,1))
y_train=np.ones(shape=(N,2))
x_train = x_train.astype('float32')
y_train = y_train.astype('int32')
#model.compile(loss=keras.losses.categorical_crossentropy,optimizer=keras.optimizers.Adam(),metrics=['accuracy'])
#model.compile(loss=keras.losses.categorical_crossentropy,optimizer=keras.optimizers.SGD(lr=0.001,momentum=0.9,decay=0.00016666,nesterov=False),metrics=['accuracy'])
np.set_printoptions(precision=4)
filelist=getFileList("./")
#shuffle(filelist)

model=load_model('/work/mlpdb/save/FOp212121train_20180709a')
model.load_weights('/work/mlpdb/save/FOp212121train.h5_20180709a')

for ifile, file in enumerate(filelist):
# print(file)
 XRAY=0
 
 if ifile < 0:
   pass
 else:
   if ifile > 2000000000:
      os.remove("/work/mlpdb/runfo")
   if not os.path.exists("/work/mlpdb/runfo"):
      print(ifile,i,file,"break at runfo exists")
      break
   if file[-3:] == ".gz":
     with gzip.open(file,'rb') as f:
       SPAC=0
       cella,cellb,cellc = 0., 0., 0.
       for line in f:
           words=line.split()
#           print(line)
           if len(words) == 0:
              pass
#           elif words[0] == b"_symmetry.Int_Tables_number":
#              if words[1] == b"19":
#                SPAC = 19
#                break
           elif words[0] == b"_symmetry.space_group_name_H-M":
#             print(line)
#             print(words[0])
#             print(words[1])
             try:
               if words[1] != b"'P" :
                 break
               else:
                 if words[2] != b"21" :
                   break
                 else:
                   if words[3] != b"21" :
                     break
                   else:
                     if words[4] == b"21" or  words[4] == b"21'"   :
                       SPAC=19
                     else:
                       break
             except:
               print(file,words)
               pass
           elif words[0] == b"_cell.length_a":
             try:
               cella = float(words[1])
             except:
               print("in _cell.lena match")
               print(words)
           elif words[0] == b"_cell.length_b":
             try:
               cellb = float(words[1])
             except:
               print("in _cell.lenb match")
               print(words)
           elif words[0] == b"_cell.length_c":
             try:
               cellc = float(words[1])
             except:
               print("in _cell.lenc match")
               print(words)
           elif line[:6] == b"1 1 1 ":
               break
             
           if SPAC == 19 and cella*cellb*cellc != 0  :
               cells = [cella,cellb,cellc]
               print(ifile,i,file)
               break

             
     if SPAC == 19 and 10. < min(cells) and max(cells) < 200.:
       pdbid=file[3:-9]
       pdbdir=file[4:6]
#         print(pdbid,pdbdir)
       pdbfile = "/work/pdbdb/"+pdbdir+"/pdb"+pdbid+".ent.gz"
       if not os.path.exists(pdbfile):
           pass
           print(ifile,i,file,"pdbfile not  exists")
       else:
         hklf='/ramdisk/tempfo0_%d.mtz'%ifile
         pdbf='/ramdisk/tempfo%d.pdb'%ifile

         in_f = gzip.GzipFile(pdbfile,'rb')
         out_f = open(pdbf,'w')
         out_f.write(str(in_f.read())[2:-1].replace("\\n","\n"))
         out_f.close()
         in_f.close()

         fobsf='/ramdisk/tempfo_%d.cif'%ifile

         Foincide = False
         in_fo = gzip.GzipFile(file,'rb')
         out_fo = open(fobsf,'w')
         strtemp=str(in_fo.read())[2:-1].replace("\\n","\n")
         if "F_meas_au" in strtemp:
             Foincide=True
         out_fo.write(strtemp)
         out_fo.close()
         in_fo.close()

         if not Foincide :
            result = 2
            print("skip no Fobs \n")
         else:
            hklfotemp='/ramdisk/tempfo_%d.mtz'%ifile
         
            cmd001="/work/ccp4/ccp4-7.0/bin/cif2mtz hklin %s hklout %s <<EOF>/ramdisk/ciflog\n"%(fobsf,hklfotemp)  
            cmd0011_2 = "END\n" +"EOF\n"

            subprocess.call(cmd001+cmd0011_2,shell=True)
            
            result=pdbtohkl(pdbf,hklf,ifile)
  
            if result != 0:
#              shutil.copyfile('/ramdisk/sfalllog','/ramdisk/sfallerr')
#              shutil.copyfile(pdbf,'/ramdisk/sfallerrpdb.pdb')
              pass
         if result == 0:
           i=i+1
           ab_pdb = False
           for iran in [0,1,2,3,4,5,6,7]:
             if ab_pdb == True:
               break
             if iran > 0 : 
                dran=10.
             else:
                dran=0.
             mapif='/ramdisk/tempfo%d_%d.map'%(i,iran)
             hklfotoranmap(hklf,hklfotemp,mapif,ifile,cells,dran)


             if os.path.exists(mapif):        
               mapb=open(mapif,'rb')
               NC,NR,NS=struct.unpack_from("III",mapb.read(),0)
   
               mapb.seek(1344,0) # 1344 = 1024+320 header + symmP212121
               mapmat=np.fromfile(mapb,np.float32,-1).reshape((NS,NR,NC,1))
   
               iNC,iNR,iNS = int(40/NC)+1, int(40/NR)+1, int(40/NS)+1
   
               mapext=np.tile(mapmat,(iNS,iNR,iNC,1))#
 
               mapnrm=mapext[:40,:40,:40,:1]

               sigma=np.std(mapnrm)
               ave=np.mean(mapnrm)
               if sigma <= 0.01:
#                 shutil.copyfile(mapif,'/ramdisk/error.map')
#                 shutil.copyfile(pdbf,'/ramdisk/error.pdb')
                 print(ifile,file,'skipped\n')
                 ab_pdb=True
                 if os.path.exists(mapif):
                    os.remove(mapif)
                 break
               mapnrmnrm = (mapnrm-ave)*0.2/sigma 
               np.clip(mapnrmnrm,-1.,1.,out=mapnrmnrm)              

               x_train[j]=mapnrmnrm
               if iran == 0:
                 y_train[j]=[1,0]
               else:
                 y_train[j]=[0,1]
#               y_train[j]=[1-iran,iran] #[1,0] correct phase, [0,1] random phase
               j=j+1

               if os.path.exists(mapif):
                  os.remove(mapif)
               if j >= N :
                 if  k % 1 == 0:
                    cur_result2=model.evaluate(x_train,y_train)
                    sumeva=0.
                    for jj in range(N):
                     

                      print(y_train[jj],end="")
                      print("  ",end="")
                      pripre=model.predict(x_train)[jj]

                      sumeva=sumeva+pripre[0]*y_train[jj][0]
                      print(pripre,end="")
                      if y_train[jj][0] == 1 and y_train[jj][1] == 0:
                         print(" correct ")
                      else:
                         print(" random  ")
                    aveeva=sumeva/N*8.     
                    print("loss= %8.3f "%cur_result2[0],"precise = %8.3f "%cur_result2[1],"eva ave = %8.3f"%aveeva)
                    logtrainf=open('/work/mlpdb/prog/log_train_fo','a')
                    logtrainf.write('i= %d '%i+'loss= %f '%cur_result2[0]+'prec= %f '%cur_result2[1]+' ave eva = %f \n'%aveeva)
                    logtrainf.close()
                 print('train_batch ',N,i)
                 j = 0
                 if aveeva > 0.0 :

                    model.train_on_batch(x_train,y_train)

                    k = k+1
                 else:
                    print("skip train \n ")
                
                 model.save('/work/mlpdb/save/FOp212121train')
                 json_string = model.to_json()
                 open('/work/mlpdb/save/FOp212121train.json','w').write(json_string)
                 model.save_weights('/work/mlpdb/save/FOp212121train.h5')
         if os.path.exists(hklf):
            os.remove(hklf)
         if os.path.exists(pdbf):
            os.remove(pdbf)
         if os.path.exists(fobsf):
            os.remove(fobsf)
         if os.path.exists(hklfotemp):
            os.remove(hklfotemp)

model.save('/work/mlpdb/save/FOp212121train')
json_string = model.to_json()
open('/work/mlpdb/save/FOp212121train.json','w').write(json_string)
model.save_weights('/work/mlpdb/save/FOp212121train.h5')
