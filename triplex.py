"""
Created on Wed May  8 13:36:46 2024
@author: Wonsang Hwang ( whwang6@mgh.harvard.edu )
"""


from IPython import get_ipython
get_ipython().magic('reset -sf')
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import cv2



def Median_filter_iw(img,inten,n_conv):
    tem=img*inten
    fil=np.ones([n_conv,n_conv])
    
    if n_conv==1:
        img_med=img
    elif n_conv%2==0:
        fil[int((n_conv//2)),int((n_conv//2))]=n_conv
    else :
        fil[int((n_conv//2)+1),int((n_conv//2)+1)]=n_conv
    
    img_f=signal.convolve2d(tem, fil, boundary='symm', mode='same')/sum(sum(np.array(fil)))
    img_inten=signal.convolve2d(inten, fil, boundary='symm', mode='same')/sum(sum(np.array(fil)))
    img_med=img_f/img_inten
    return img_med
def triplex(c1,c2,c3,g,s):
    g=np.array(g)
    s=np.array(s)
    a1=((c2[1]-np.array(s))/(c2[0]-np.array(g)))
    b1=a1*(c2[0])-c2[1]
    a2=((c1[1]-c3[1])/(c1[0]-c3[0]))
    b2=a2*(c1[0])-c1[1]
        
    coe1=[a1,-b1]
    coe2=[a2,-b2]

    coe1=np.array(coe1)

    g_int=(coe2[1]-coe1[1])/(coe1[0]-coe2[0])
    s_int=g_int*coe2[0]+coe2[1]
    dis1=np.sqrt((c2[0]-g)**2+(c2[1]-s)**2)
    dis2=np.sqrt((c2[0]-g_int)**2+(c2[1]-s_int)**2) 
    dis3=np.sqrt((c1[0]-c3[0])**2+(c1[1]-c3[1])**2) 
    dis4=np.sqrt((c1[0]-g_int)**2+(c1[1]-s_int)**2) 

    f13=dis1/dis2
    f13[f13>1]=1
    f2=1-f13

    f3=(dis4/dis3)*f13
    f1=(1-dis4/dis3)*f13

    f1=f1/(f1+f2+f3)
    f2=f2/(f1+f2+f3)
    f3=f3/(f1+f2+f3)
    return f1,f2,f3



data_dir='C:\\Example_triplex\\' # Directory for your data 
i = np.load(data_dir + 'inten.npy') #  load intensity data  
g = np.load(data_dir + 'g.npy') # load g data
s = np.load(data_dir + 's.npy') # load s data

f=0.05 # modulation frequency
w=2*np.pi*f # angular modulation frequency
n_av=5 # median filter size 

g_m=Median_filter_iw(g,i,n_av) # median filter applied
s_m=Median_filter_iw(s,i,n_av) # median filter applied
fl_p=s_m/g_m/w # fluorescence lifetime at the modulation frequency 

# put your own reference phasors below

c1= [0.912,0.286] # reference phasor 1
c2= [ 0.648,0.469] # reference phasor 2
c3= [ 0.442, 0.481] # reference phasor 3

f1,f2,f3=triplex(c1,c2,c3,g_m,s_m) # multiplexing three components

f1[np.isnan(f1)]=0 # make nan value zero 
f2[np.isnan(f2)]=0 # make nan value zero
f3[np.isnan(f3)]=0 # make nan value zero

i_1=f1*i # multiplexed intensity of channel 1
i_2=f2*i # multiplexed intensity of channel 2
i_3=f3*i # multiplexed intensity of channel 3

i_composite=cv2.merge([(i_1/np.max(i_1)),(i_2/np.max(i_2)),(i_3/np.max(i_3))]) # make a RGB composite image. red: channel_1 , green: channel_2 , blue: channel_3
plt.imshow(i_composite) # composite image plot
plt.title("R: Ch1   G: Ch2   B: Ch3")
