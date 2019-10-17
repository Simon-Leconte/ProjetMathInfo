import autograd
from autograd import numpy as np

# Attention : La fonction autograd.grad renvoit une fonction qui prend en
# paramètre un float et non un int

def grad(f,x,y):
    g=autograd.grad
    return np.r_[g(f,0)(x,y), g(f,1)(x,y)]

def derive(f,x):
    g=autograd.grad(f)
    return g(x)

def find_seed(g,c=0.0,eps=2**(-26)):
    """ Entrée :
        Une fonction de deux variables g
        Une valeur c telle qu'il existe t tel que g(0,t)=c
        Une précision epsilon sur la valeur de t renvoyée
        Sortie :
        Une valeur de t telle que g(0,t)=c"""
    def f_g(y):
        return g(0.0,y)
    y1=0.0
    y2=1.0
    while abs(y1-y2)>eps :
        print(y1)
        pente = derive(f_g,y1)
        print("pente : ",pente)
        y2=y1
        y1=y1-f_g(y1)/pente
    return y1

def carre(x,y):
    return x**2+y**2+y+0.5
def carre1(x):
    return x**2