
##########  TIPE  ##########


## TRIANGULATION NAÏVE

from math import *
import time
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
import numpy as np


poly = [[0,0],[0.5,-1],[1.5,-0.2],[2,-0.5],[2,0],[1.5,1],[0.3,0],[0.5,1]]


plt.plot(np.array(poly+[poly[0]])[:,0],np.array(poly+[poly[0]])[:,1])


def ind_voisin(s,degre,n): #Renvoie l'indice du s+degré ième sommet.
    return (s+degre)%n

print(ind_voisin(6,2,8))

def produit_vec(S1,S2,M):   #Renvoie Vec(S1S2) ∧ Vec(S1M)
    V1=(S2[0]-S1[0],S2[1]-S1[1])
    V2=(M[0]-S1[0],M[1]-S1[1])
    return(V1[0]*V2[1] - V1[1]*V2[0])

print(produit_vec(poly[0],poly[1],poly[2]))

def triangle(polygone,indices):
    return([polygone[indices[0]],polygone[indices[1]],polygone[indices[2]]])

def point_dans_triangle(polygone,indices,M):    #Renvoie True si le point M est dans le triangle
    T=triangle(polygone,indices)
    S0=T[0]
    S1=T[1]
    S2=T[2]
    p1=produit_vec(S0,S1,M)
    p2=produit_vec(S1,S2,M)
    p3=produit_vec(S2,S0,M)
    return((p1>0 and p2>0 and p3>0) or (p1<0 and p2<0 and p3<0))

print(point_dans_triangle(poly,[0,1,2],[1,-0.5]))

def sommet_distance_max(polygone,indices): #Renvoie l'indice du sommet dans le triangle ABC, s'il existe, qui se situe le plus loin de la droite AC.
    #"indices" correspond aux indices des sommets du triangle
    n=len(polygone)
    maxi=-1
    ind_max=-1
    T=triangle(polygone,indices)
    for i in range(n):
        M=polygone[i]
        if not(i in indices) and point_dans_triangle(polygone,indices,M):
            x = abs(produit_vec(T[0],T[2],M))
            if x>maxi:
                ind_max=i
                maxi=x
    if ind_max==-1: return(None)
    else: return(ind_max)


print(sommet_distance_max(poly,[0,3,4]))


def sommet_gauche(polygone):        #Obligé d'utiliser une fonction comme celle-ci pour garantir une certaine forme de convexité (on ne traite que des triangles qui sont contenus dans le polygone)
    n=len(polygone)
    if n<3: return('problème sur le nbre de sommets')
    xg = polygone[0][0]
    ig = 0
    for i in range(1,n):
        if polygone[i][0] < xg:
            xg = polygone[i][0]
            ig = i
    return ig

def nouveau_polygone(polygone,i_d,i_f):      #Sert à simplifier l'écriture de la fonction récursive finale
    n=len(polygone)
    if n==3:
        return(polygone)
    else:
        new_p=[]
        i=i_d
        while i != i_f :
            new_p.append(polygone[i])
            i=ind_voisin(i,1,n)
        new_p.append(polygone[i_f])
        return(new_p)

print(poly)
print(nouveau_polygone(poly,1,3))
print('-----------------------------------------------------')


## Fonction finale

def trianguler(polygone):
    n0 = len(polygone)
    if n0 < 3:
        return("Trop peu de sommets")
    else:
        def trianguler_rec(polygone,liste_triangles):
            n=len(polygone)
            i1=sommet_gauche(polygone)
            i0=ind_voisin(i1,-1,n)
            i2=ind_voisin(i1,1,n)
            T=triangle(polygone,[i0,i1,i2])             #On traite CE triangle
            i=sommet_distance_max(polygone,[i0,i1,i2])
            if i==None:    # ie: aucun sommet du polygone contenu dans le traingle
                liste_triangles.append(T)
                poly1=nouveau_polygone(polygone,i2,i0)
                if len(poly1)==3:
                    liste_triangles.append(poly1)
                else:      #On procède récursivement sur les sommets qui restent
                    trianguler_rec(poly1,liste_triangles)
            else:          # ie: il y a (au moins) un sommet contenu dans le triangle
                poly1=nouveau_polygone(polygone,i1,i)
                poly2=nouveau_polygone(polygone,i,i1)
                if len(poly1)==3:
                    liste_triangles.append(poly1)
                else:
                    trianguler_rec(poly1,liste_triangles)
                if len(poly2)==3:
                    liste_triangles.append(poly2)
                else:
                    trianguler_rec(poly2,liste_triangles)
                #On procède récursivement sur les deux polygones ainsi créés
            return(liste_triangles)

        return(trianguler_rec(polygone,[]))

print(trianguler(poly))


print()
print('-------------------------------------')


def draw_liste_triangles(liste_triangles):
    fig,ax = plt.subplots()
    patches = []
    for triangle in liste_triangles:
        patches.append(Polygon(triangle))
    p = PatchCollection(patches, alpha=1.0)
    colors = 100*np.random.rand(len(patches))
    p.set_array(np.array(colors))
    ax.add_collection(p)


poly = [[0,0],[0.5,-1],[1.5,-0.2],[2,-0.5],[2,0],[1.5,1],[0.3,0],[0.5,1]]
plt.plot(np.array(poly+[poly[0]])[:,0],np.array(poly+[poly[0]])[:,1],color='red')

liste_triangles = trianguler(poly)
draw_liste_triangles(liste_triangles)
plt.axis([0,2,-2,2])

plt.show()





##### Test affichage 2 images #####

## Cas simple ##


poly = [[0,0],[0.5,-1],[1.5,-0.2],[2,-0.5],[2,0],[1.5,1],[0.3,0],[0.5,1]]

f = plt.figure()
f.add_subplot(1,2,1)
plt.plot(np.array(poly+[poly[0]])[:,0],np.array(poly+[poly[0]])[:,1],color='black')

f.add_subplot(1,2,2)

l_triangles = trianguler(poly)

for T in l_triangles:
    plt.plot(np.array(T+[T[0]])[:,0],np.array(T+[T[0]])[:,1],color='black')


plt.show()

## Cas complexe ##



poly = [[-1.5,1.94],[-2.46,-4.3],[3.18,-1.8],[5.7,-6.84],[10,-4],[4.9,-1.56],[12,-1],[5.2,0.42],[11.86,2.94],[5,2],[4.62,4.04],[3.92,2.28],[1.42,2.48],[-0.76,3.72],[0,1.56],[3.28,0.92],[1.82,-1]]

f = plt.figure()
f.add_subplot(1,2,1)
plt.plot(np.array(poly+[poly[0]])[:,0],np.array(poly+[poly[0]])[:,1],color='black')

f.add_subplot(1,2,2)

l_triangles = trianguler(poly)

for T in l_triangles:
    plt.plot(np.array(T+[T[0]])[:,0],np.array(T+[T[0]])[:,1],color='black')


plt.show()





##### Triangles voisins #####

poly = [[0,0],[0.5,-1],[1.5,-0.2],[2,-0.5],[2,0],[1.5,1],[0.3,0],[0.5,1]]

#poly = [[-1.5,1.94],[-2.46,-4.3],[3.18,-1.8],[5.7,-6.84],[10,-4],[4.9,-1.56],[12,-1],[5.2,0.42],[11.86,2.94],[5,2],[4.62,4.04],[3.92,2.28],[1.42,2.48],[-0.76,3.72],[0,1.56],[3.28,0.92],[1.82,-1]]

l_triangles = trianguler(poly)


def voisin(T1,T2):
    nb_communs=0
    for x in T1:
        if x in T2:
            nb_communs+=1
    return(nb_communs==2)

print(l_triangles)
print(voisin(l_triangles[1],l_triangles[2]))


def liste_adj(l):
    n=len(l)
    adj=[[] for _ in range(n)]
    for i in range(n):
        for j in range(i+1,n):
            if voisin(l[i],l[j]):
                adj[i].append(j)
                adj[j].append(i)
    return(adj)

l=liste_adj(l_triangles)
print(l)



##### Parcours en largeur #####

def dans_triangle(T,M): #Vérifie si M est dans le triangle T (comprend les bords)
    S0=T[0]
    S1=T[1]
    S2=T[2]
    p1=produit_vec(S0,S1,M)
    p2=produit_vec(S1,S2,M)
    p3=produit_vec(S2,S0,M)
    return((p1>=0 and p2>=0 and p3>=0) or (p1<=0 and p2<=0 and p3<=0))

def trouve_depart_arrivee(l,d,a):   #l=l_triangles  #Permet de trouver le triangle de départ et d'arrivée
    n=len(l)
    i_d,i_a=0,0
    for i in range(n):
        if dans_triangle(l[i],d):
            i_d=i
        if dans_triangle(l[i],a):
            i_a=i
    return(i_d,i_a)

def centre_triangle(T):
    t_moy=np.round(np.mean(T,axis=0),2)
    return([t_moy[0],t_moy[1]])


def parcours_en_largeur(l_adj,depart):
    n=len(l_adj)
    file=[depart]
    c=["b" for _ in range(n)]
    p=[-1 for _ in range(n)]
    def traite_fils(u,l_voisins):
        for i in l_voisins:
            if c[i]=="b":
                c[i]="g"
                p[i]=u
                file.append(i)
    def traite(u):
        traite_fils(u,l_adj[u])
        c[u]="n"

    while file != []:
        traite(file.pop(0))
    return(p)



##### Algorithme final #####

poly = [[0,0],[0.5,-1],[1.5,-0.2],[2,-0.5],[2,0],[1.5,1],[0.3,0],[0.5,1]]

#poly = [[-1.5,1.94],[-2.46,-4.3],[3.18,-1.8],[5.7,-6.84],[10,-4],[4.9,-1.56],[12,-1],[5.2,0.42],[11.86,2.94],[5,2],[4.62,4.04],[3.92,2.28],[1.42,2.48],[-0.76,3.72],[0,1.56],[3.28,0.92],[1.82,-1]]

depart=[0.3,0.5]
arrivee=[2,-0.3]

def PCC_Triangulation(P,depart,arrivee):
    l_T=trianguler(P)
    D,A=trouve_depart_arrivee(l_T,depart,arrivee)
    l_adj=liste_adj(l_T)
    p=parcours_en_largeur(l_adj,D)

    i=A
    ordre_t=[i]
    while ordre_t[0] != D:
        ordre_t = [p[i]] + ordre_t
        i=p[i]

    chemin=[]
    for i in ordre_t:
        T=l_T[i]
        x=centre_triangle(T)
        chemin.append(x)
    chemin=[depart]+chemin+[arrivee]
    return(chemin)

chemin=PCC_Triangulation(poly,depart,arrivee)
print(chemin)


##### Tracé graphique #####

#poly = [[0,0],[0.5,-1],[1.5,-0.2],[2,-0.5],[2,0],[1.5,1],[0.3,0],[0.5,1]]
#depart=[0.3,0.5]
#arrivee=[2,-0.3]

poly = [[-1.5,1.94],[-2.46,-4.3],[3.18,-1.8],[5.7,-6.84],[10,-4],[4.9,-1.56],[12,-1],[5.2,0.42],[11.86,2.94],[5,2],[4.62,4.04],[3.92,2.28],[1.42,2.48],[-0.76,3.72],[0,1.56],[3.28,0.92],[1.82,-1]]
depart=[0.,3]
arrivee=[10,-0.9]

t1=time.time()

f = plt.figure()


f.add_subplot(1,2,1)

plt.plot(np.array(poly+[poly[0]])[:,0],np.array(poly+[poly[0]])[:,1],color='black')
plt.plot(depart[0],depart[1],marker='o')
plt.plot(arrivee[0],arrivee[1],marker='o')


f.add_subplot(1,2,2)

l_triangles = trianguler(poly)
for T in l_triangles:
    plt.plot(np.array(T+[T[0]])[:,0],np.array(T+[T[0]])[:,1],color='black')

chemin=PCC_Triangulation(poly,depart,arrivee)
plt.plot(np.array(chemin)[:,0],np.array(chemin)[:,1],color='red')

plt.plot(depart[0],depart[1],marker='o')
plt.plot(arrivee[0],arrivee[1],marker='o')

t2=time.time()

print('Temps de calcul :',t2-t1)


plt.show()

##### Test temps #####  Question : Faut-il inclure le tracé dans le calcul du temps ?

t1=time.time()
chemin=PCC_Triangulation(poly,depart,arrivee)
print(chemin)
t2=time.time()

print('Temps de calcul :',t2-t1)






##### Triangulation de Delaunay #####

import random as r
from itertools import combinations

points = [[r.randint(-500,500)/100,r.randint(-500,500)/100] for _ in range(50)]
poly = [[-1.5,1.94],[-2.46,-4.3],[3.18,-1.8],[5.7,-6.84],[10,-4],[4.9,-1.56],[12,-1],[5.2,0.42],[11.86,2.94],[5,2],[4.62,4.04],[3.92,2.28],[1.42,2.48],[-0.76,3.72],[0,1.56],[3.28,0.92],[1.82,-1]]

print(points)

def Delaunay(ListePoints):
    ListeTriangle = list(combinations(ListePoints,3))
    Triangulation=[]
    for T in ListeTriangle:
        bool=True
        ListD=[]

        A1=2*(T[1][0]-T[0][0])
        B1=2*(T[1][1]-T[0][1])
        C1=T[0][0]**2 + T[0][1]**2 - T[1][0]**2 - T[1][1]**2

        A2=2*(T[2][0]-T[1][0])
        B2=2*(T[2][1]-T[1][1])
        C2=T[1][0]**2 + T[1][1]**2 - T[2][0]**2 - T[2][1]**2

        X=(B1*C2-B2*C1) / (A1*B2-B1*A2)
        Y=(A2*C1-A1*C2) / (A1*B2-B1*A2)

        R=(X-T[0][0])**2 + (Y-T[0][1])**2

        for P in ListePoints:
            if not(P in T):
                distance=(P[0]-X)**2+(P[1]-Y)**2
                ListD.append(distance)
                for D in ListD:
                    if D<=R:
                        bool=False
                        break
        if bool :
            Triangulation.append(T)
    return(Triangulation)



f = plt.figure()

f.add_subplot(1,2,1)

l_triangles = trianguler(poly)
for T in l_triangles:
    plt.plot(np.array(T+[T[0]])[:,0],np.array(T+[T[0]])[:,1],color='black')

f.add_subplot(1,2,2)


l_triangles=Delaunay(poly)
for T in l_triangles:
    plt.plot(np.array(list(T)+[T[0]])[:,0],np.array(list(T)+[T[0]])[:,1],color='black')

plt.show()





##### Test Gauche,droite,haut,bas #####

#poly = [[0,0],[0.5,-1],[1.5,-0.2],[2,-0.5],[2,0],[1.5,1],[0.3,-0.1],[0.5,1]]
poly = [[-1.5,1.94],[-2.46,-4.3],[3.18,-1.8],[5.7,-6.84],[10,-4],[4.9,-1.56],[12,-1],[5.2,0.42],[11.86,2.94],[5,2],[4.62,4.04],[3.92,2.28],[1.42,2.48],[-0.76,3.72],[0,1.56],[3.28,0.92],[1.82,-1]]

def distance_2points(P1,P2):
    return(np.round(sqrt((P2[0]-P1[0])**2 + (P2[1]-P1[1])**2),2))

def distance(l):
    d=0
    for T in l:
        for i in range(3):
            d+=distance_2points(T[i-1],T[i])
    return(np.round(d,2))


def sommet_direction(polygone,direction):        #Obligé d'utiliser une fonction comme celle-ci pour garantir une certaine forme de convexité (on ne traite que des triangles qui sont contenus dans le polygone)
    n=len(polygone)
    if n<3:
        return('problème sur le nbre de sommets')
    if not(direction in ['haut','bas','gauche','droite']):
        return('problème dans la notation de la direction')
    if direction == 'gauche':
        xg = polygone[0][0]
        ig = 0
        for i in range(1,n):
            if polygone[i][0] < xg:
                xg = polygone[i][0]
                ig = i
        return(ig)
    if direction == 'droite':
        xd = polygone[0][0]
        id = 0
        for i in range(1,n):
            if polygone[i][0] > xd:
                xd = polygone[i][0]
                id = i
        return(id)
    if direction == 'bas':
        yb = polygone[0][1]
        ib = 0
        for i in range(1,n):
            if polygone[i][1] < yb:
                yb = polygone[i][1]
                ib = i
        return(ib)
    if direction == 'haut':
        yh = polygone[0][1]
        ih = 0
        for i in range(1,n):
            if polygone[i][1] > yh:
                yh = polygone[i][1]
                ih = i
        return(ih)


def trianguler_opti(polygone):
    n0 = len(polygone)
    if n0 < 3:
        return("Trop peu de sommets")
    else:
        def trianguler_rec(polygone,liste_triangles,direction):
            n=len(polygone)
            i1=sommet_direction(polygone,direction)
            i0=ind_voisin(i1,-1,n)
            i2=ind_voisin(i1,1,n)
            T=triangle(polygone,[i0,i1,i2])             #On traite CE triangle
            i=sommet_distance_max(polygone,[i0,i1,i2])
            if i==None:    # ie: aucun sommet du polygone contenu dans le traingle
                liste_triangles.append(T)
                poly1=nouveau_polygone(polygone,i2,i0)
                if len(poly1)==3:
                    liste_triangles.append(poly1)
                else:      #On procède récursivement sur les sommets qui restent
                    trianguler_rec(poly1,liste_triangles,direction)
            else:          # ie: il y a (au moins) un sommet contenu dans le triangle
                poly1=nouveau_polygone(polygone,i1,i)
                poly2=nouveau_polygone(polygone,i,i1)
                if len(poly1)==3:
                    liste_triangles.append(poly1)
                else:
                    trianguler_rec(poly1,liste_triangles,direction)
                if len(poly2)==3:
                    liste_triangles.append(poly2)
                else:
                    trianguler_rec(poly2,liste_triangles,direction)
                #On procède récursivement sur les deux polygones ainsi créés
            return(liste_triangles)

        triangulation=trianguler_rec(polygone,[],'gauche')
        d=distance(triangulation)
        i=2
        choix='gauche'
        for direction in ['droite','bas','haut']:
            l_t=trianguler_rec(polygone,[],direction)
            dd=distance(l_t)
            if dd<d:
                triangulation=l_t
                d=dd
                choix=direction
            i+=1
        return(triangulation,choix)


def PCC_Triangulation_opti(P,depart,arrivee):
    l_T=trianguler_opti(P)[0]
    D,A=trouve_depart_arrivee(l_T,depart,arrivee)
    l_adj=liste_adj(l_T)
    p=parcours_en_largeur(l_adj,D)

    i=A
    ordre_t=[i]
    while ordre_t[0] != D:
        ordre_t = [p[i]] + ordre_t
        i=p[i]

    chemin=[]
    for i in ordre_t:
        T=l_T[i]
        x=centre_triangle(T)
        chemin.append(x)
    chemin=[depart]+chemin+[arrivee]
    return(chemin)

##### Tracé graphique "trianguler" optimisée #####


#poly = [[0,0],[0.5,-1],[1.5,-0.2],[2,-0.5],[2,0],[1.5,1],[0.3,0],[0.5,1]]
#depart=[0.3,0.5]
#arrivee=[2,-0.3]

poly = [[-1.5,1.94],[-2.46,-4.3],[3.18,-1.8],[5.7,-6.84],[10,-4],[4.9,-1.56],[12,-1],[5.2,0.42],[11.86,2.94],[5,2],[4.62,4.04],[3.92,2.28],[1.42,2.48],[-0.76,3.72],[0,1.56],[3.28,0.92],[1.82,-1]]
depart=[-0.5,3.41]
arrivee=[5.2,-1.8]

t1=time.time()

f = plt.figure()


f.add_subplot(1,2,1)

plt.plot(np.array(poly+[poly[0]])[:,0],np.array(poly+[poly[0]])[:,1],color='black')
plt.plot(depart[0],depart[1],marker='o')
plt.plot(arrivee[0],arrivee[1],marker='o')


f.add_subplot(1,2,2)

l_triangles,choix= trianguler_opti(poly)
for T in l_triangles:
    plt.plot(np.array(T+[T[0]])[:,0],np.array(T+[T[0]])[:,1],color='black')

chemin=PCC_Triangulation_opti(poly,depart,arrivee)
plt.plot(np.array(chemin)[:,0],np.array(chemin)[:,1],color='red')

plt.plot(depart[0],depart[1],marker='o')
plt.plot(arrivee[0],arrivee[1],marker='o')

t2=time.time()

print('Choix de la direction de parcours :',choix)
print('Temps de calcul :',t2-t1)


plt.show()











##### DIJKSTRA #####

poly = [[0,0],[0.5,-1],[1.5,-0.2],[2,-0.5],[2,0],[1.5,1],[0.3,0],[0.5,1]]
depart=[0.3,0.5]
arrivee=[2,-0.3]

#poly = [[-1.5,1.94],[-2.46,-4.3],[3.18,-1.8],[5.7,-6.84],[10,-4],[4.9,-1.56],[12,-1],[5.2,0.42],[11.86,2.94],[5,2],[4.62,4.04],[3.92,2.28],[1.42,2.48],[-0.76,3.72],[0,1.56],[3.28,0.92],[1.82,-1]]
#depart=[-0.5,3.41]
#arrivee=[5.2,-1.8]

plt.plot(np.array(poly+[poly[0]])[:,0],np.array(poly+[poly[0]])[:,1],color='black')
plt.plot(depart[0],depart[1],marker='o')
plt.plot(arrivee[0],arrivee[1],marker='o')

plt.show()

def orientation(X,Y,M):
    p=produit_vec(X,Y,M)
    if p>0: return(1)
    elif p==0: return(0)
    else: return(-1)

def croisement(X,Y,P1,P2):
    if X==P1 or X==P2 or X==P1 or X==P1 or orientation(P1,P2,X)==0 or orientation(P1,P2,Y)==0:
        return False
    if orientation(P1,X,P2)==orientation(P1,X,Y) and orientation(P1,Y,X)==orientation(P1,Y,P2) and orientation(X,Y,P1) != orientation(X,Y,P2):
        return True
    else:
        return False


print(croisement(poly[1],poly[3],poly[2],poly[3]))

print('--------------------------')

def croisement_polygone(X,Y,poly): #renvoie True si [XY] ne passe pas dans le polygone
    for k in range(len(poly)):
        if croisement(X,Y,poly[k-1],poly[k]):
            return True
    return False

print(croisement_polygone(poly[0],poly[6],poly))


def dans_polygone(i,j,poly):
    n=len(poly)
    if abs(i-j)==1 or (i>=n or j>=n):
        return True
    else:
        X=poly[i]
        Y=poly[j]
        P0=poly[i-1]
        P1=poly[(i+1)%n]
        if orientation(P0,X,P1)== 1:
            return (orientation(P0,X,P1) == orientation(P0,X,Y) and orientation(P1,X,P0) == orientation(P1,X,Y))
        else:
            return (orientation(P0,X,P1) != orientation(P0,X,Y) or orientation(P1,X,P0) != orientation(P1,X,Y))


def graphe(poly,depart,arrivee):
    S=poly+[depart]+[arrivee]
    n=len(S)
    g=np.zeros((n,n))
    for i in range(n):
        for j in range(i+1,n):
            if not(croisement_polygone(S[i],S[j],poly)) and dans_polygone(i,j,poly):
                g[i,j]=distance_2points(S[i],S[j])
                g[j,i]=g[i,j]
    return(g)

print(graphe(poly,depart,arrivee))


##





















