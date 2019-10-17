
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
    return y
find_seed(g,0.31415926)