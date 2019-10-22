import autograd
from autograd import numpy as np
import matplotlib.pyplot as plt


def find_seed(g,c=0.0,eps=2**(-26)):
    a=0.0
    b=1.0
    if not(g(0,b)<c<g(0,a) or g(0,a)<c<g(0,b)):
        return None
    while abs(a-b)>eps:
        milieu = (a+b)/2
        if (g(0.0,milieu)<c) ^ (g(0,b)<g(0,a)):
            a=milieu
        else:
            b=milieu
    t=a
    return t

def f(x,y):
    return x**2+(y-0.2)**2

def delta_normalisation(l,delta):
    """ Cette fonction prend en paramètre un gradient, et modifie sa norme de
    telle sorte qu'elle vaille delta en sortie """
    norme = np.sqrt(l[0]**2+l[1]**2)
    return l*delta/norme
def grad(f,x,y):
    """Cette fonction renvoit le gradient de f en (x,y)"""
    g=autograd.grad
    return np.r_[g(f,0)(x,y),g(f,1)(x,y)]
def simple_contour(f,c=0.0,delta = 0.01,eps = 2**(-26)):
    #Initialisation du tracé avec find_seed
    x0 = 0.0
    y0 = find_seed(f,c,eps)
    print(y0)
    if y0==None:
        return [],[]
    tabX = [x0]
    tabY = [y0]
    #Première étape, nécessairement vers les x croissants
    #Calcul du gradient
    gradient = grad(f,x0,y0)
    grad_utile = delta_normalisation(gradient,delta)
    # Il faut distinguer le cas grad_utile[1]>0 et le cas grad_utile[1]<0 pour
    # savoir quelle normale au gradient choisir
    b = grad_utile[1]
    a = grad_utile[0]
    if b>0:
        tabX.append(x0+b)
        tabY.append(y0-a)
    else:
        tabX.append(x0-b)
        tabY.append(y0+a)
    #La première étape est terminée
    #Tant que l'on est encore dans la case [0,1]², on continue à prolonger
    while 0<=tabX[-1]<=1 and 0<=tabY[-1]<=1:
        #On commence par calculer le gradient normalisé au dernier point
        gradient = grad(f,tabX[-1],tabY[-1])
        grad_utile = delta_normalisation(gradient,delta)
        b = grad_utile[1]
        a = grad_utile[0]
        deplacementPrecedent = (tabX[-2]-tabX[-1],tabY[-2]-tabY[-1])
        #On calcule le produit vectoriel entre le déplacement précédent et le
        #gradient actuel pour déterminer la direction de déplacement à l'étape
        #suivante
        vectoriel = deplacementPrecedent[0]*b-deplacementPrecedent[1]*a
        #Selon le signe du produit vectoriel, on détermine quelle normale au
        #gradient choisir
        if vectoriel > 0:
            tabX.append(tabX[-1]-b)
            tabY.append(tabY[-1]+a)
        else:
            tabX.append(tabX[-1]+b)
            tabY.append(tabY[-1]-a)
    return tabX,tabY

x,y=simple_contour(f,c=0.5)
plt.clf()
plt.plot(x,y)
plt.show()
