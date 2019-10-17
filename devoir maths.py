import autograd
from autograd import numpy as np
import matplotlib as plt


def find_seed(g,c=0.0,eps=2**(-26)):
    a=0.0
    b=1.0
    if c<g(0,a) or g(0,b)<c :
        return None
    while abs(a-b)>eps:
        milieu = (a+b)/2
        if (g(0.0,milieu)<c) ^ (g(0,b)<g(0,a)):
            a=milieu
        else:
            b=milieu
    t=a
    return t

#Test
def g(x,y):
    return float(x**2+y**2)
find_seed(g,0.31415926)

def grad_f(f,x,y):
    g=autograd.grad
    return np.r_[g(f,0)(x,y),g(f,1)(x,y)]

def h(x,y):
    return x+y
print(grad_f(h,0.0,0.0))

def simple_contour(f,c=0.0,delta=0.01):
    x=np.array([])
    y=np.array([])
    y0=find_seed(f,c=0.0,eps=2**(-26))
    if y0==None :
        return x,y
    x=x+np.array([0.0])
    y=y+np.array([y0])
    t=grad_f(f,0.0,float(y0))
    print(t)
    a=t[0]
    b=t[1]
    norme=(a**2+b**2)**0.5
    if b<0 :
        print("1")
        dir_x=f*np.array[b]
        dir_y=f*np.array[-a]
        x=x+dir_x
        y=y+dir_y
    else:
        print(2)
        dir_x=f*np.array[-b]
        dir_y=f*np.array[a]
        x=x+dir_x
        y=y+dir_y
    while x[-1]<=1 and y[-1]<=1 and x[-1]>=1 and y[-1]>=1  :
        print(3)
        t=grad_f(f,x[-1],y[1])
        a=t[0]
        b=t[1]
        f=delta/((a**2+b**2)**0.5)
        dir_ant=(x[-1]-x[-2])*b-a*(y[-1]-y[-2])
        if dir_ant>0 :
            print(4)
            dir_x=f*np.array[b]
            dir_y=f*np.array[-a]
            x=x+dir_x
            y=y+dir_y
        else:
            print(5)
            dir_x=f*np.array[-b]
            dir_y=f*np.array[a]
            x=x+dir_x
            y=y+dir_y
    return x,y
x,y=simple_contour(g,0.5)
plt.plot(x,y)
plt.show()
