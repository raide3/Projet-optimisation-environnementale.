import numpy as np




taille_cent=int(input("entrer taille couche entrée"))
taille_ccach=int(input("entrer taille couche cachée"))
taille_csort=int(input("entrer taille couche sortie"))
T=int(input("entrer taille de la sequence (longueur)"))
# ligne c'est taille couche entrée et colonne c'est T
U=np.random.uniform(-1,1,size=(taille_ccach,taille_cent))
V=np.random.uniform(-1,1,size=(taille_ccach,taille_ccach))
W=np.random.uniform(-1,1,size=(taille_csort,taille_ccach))
Bh=np.random.uniform(-1,1,size=(taille_ccach,1))
Bo=np.random.uniform(-1,1,size=(taille_csort,1))

lstX=[]
for t in range(T):
    Xt=np.random.rand(taille_cent)
    lstX.append(Xt)

vtanh=np.vectorize(np.tanh)

X=np.array(lstX)
Hp=np.zeros((taille_ccach,1))
for i in range (10):
    print(t)
    comp1=U@Xt
    comp2=V@Hp
    add=comp1+comp2+Bh
    Ht=vtanh(add)
    Hp=Ht

#output layre
ZT=np.dot(W,Ht)+Bo

# la sortie de la couche avec la fonction softmax
vexp=np.vectorize(np.exp)
sortie= vexp(ZT)
entre=sortie/np.sum(sortie)

for i in range(len(ZT)):
    print(f'output i{round(sortie[i,0],2)}')


#print(conv(taille_image,taille_filtre))
#print(pooling(conv(taille_imag5
# 4
# e,taille_filtre)))
