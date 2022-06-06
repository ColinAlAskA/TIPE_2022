
##########  TIPE - POLYGONE ##########

from math import *
import time
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
import numpy as np

def agrandir1():
    mngr = plt.get_current_fig_manager()
    mngr.window.setGeometry(400,50,1100,1000)
    #f.set_tight_layout(True)
    plt.tight_layout(pad=0.5)

def agrandir2():
    mngr = plt.get_current_fig_manager()
    mngr.window.setGeometry(50,50,1800,1000)
    #f.set_tight_layout(True)
    plt.tight_layout(pad=0.1)

## Les polygones ##


poly = [[0,0],[0.5,-1],[1.5,-0.2],[2,-0.5],[2,0],[1.5,1],[0.3,0],[0.5,1]]
depart=[0.3,0.5]
arrivee=[2,-0.3]


poly = [[-1.5,1.94],[-2.46,-4.3],[3.18,-1.8],[5.7,-6.84],[10,-4],[4.9,-1.56],[12,-1],[5.2,0.42],[11.86,2.94],[5,2],[4.62,4.04],[3.92,2.28],[1.42,2.48],[-0.76,3.72],[0,1.56],[3.28,0.92],[1.82,-1]]
depart=[-0.5,3.41]
arrivee=[-1.58,-0.44]


poly = [[-3.37,15.03],[-9.44,11.54],[-4.31,11.75],[-8.45,6.59],[-9.61,2.98],[-16.96,-1.94],[-12.48,-0.72],[-6.99,-4.09],[0.86,0.86],[-2.17,6.39],[3.57,3.73],[3.07,-2.09],[0.78,-5.25],[5.94,-2.26],[7.13,4.71],[12.76,-2.59],[17.08,3.52],[11.37,2.18],[26.93,13.4],[6.48,6.39],[1.79,5.23],[0.55,8.51],[8.6,15.7],[14.79,12.7],[16.79,14.95],[12.46,16.65],[1.2,16.4],[5.44,15.32],[2.03,15.28],[-6.20,1.19],[-6.02,4.49],[-0.71,15.24],[-2.29,16.86]]
depart = [-6.70,12.70]
arrivee = [15,1.64]







##### TRIANGULATION NAÏVE #####


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


# def draw_liste_triangles(liste_triangles):
#     fig,ax = plt.subplots()
#     patches = []
#     for triangle in liste_triangles:
#         patches.append(Polygon(triangle))
#     p = PatchCollection(patches, alpha=1.0)
#     colors = 100*np.random.rand(len(patches))
#     p.set_array(np.array(colors))
#     ax.add_collection(p)
# draw_liste_triangles(liste_triangles)


plt.plot(np.array(poly+[poly[0]])[:,0],np.array(poly+[poly[0]])[:,1],color='black')

liste_triangles = trianguler(poly)
plt.axis([-20,30,-7,20])
plt.plot(depart[0],depart[1],marker='o',c='g',markersize=10)
plt.plot(arrivee[0],arrivee[1],marker='o',c='orange',markersize=10)

agrandir1()
plt.show()





##### Test affichage #####

f = plt.figure()
f.add_subplot(1,2,1)
plt.plot(np.array(poly+[poly[0]])[:,0],np.array(poly+[poly[0]])[:,1],color='black')

f.add_subplot(1,2,2)

l_triangles = trianguler(poly)

for T in l_triangles:
    plt.plot(np.array(T+[T[0]])[:,0],np.array(T+[T[0]])[:,1],color='blue')
plt.plot(np.array(poly+[poly[0]])[:,0],np.array(poly+[poly[0]])[:,1],color='black')

agrandir2()
plt.show()


##### Triangles voisins #####


def voisin(T1,T2):
    nb_communs=0
    for x in T1:
        if x in T2:
            nb_communs+=1
    return(nb_communs==2)

def liste_adj(l):
    n=len(l)
    adj=[[] for _ in range(n)]
    for i in range(n):
        for j in range(i+1,n):
            if voisin(l[i],l[j]):
                adj[i].append(j)
                adj[j].append(i)
    return(adj)



##### Parcours en largeur (Breadth First Search) #####


def parcours_en_largeur(l_adj,depart):
    n=len(l_adj)
    file=[depart]
    c=["blanc" for _ in range(n)]
    p=[-1 for _ in range(n)]
    def traite_fils(u,l_voisins):
        for i in l_voisins:
            if c[i]=="blanc":
                p[i]=u
                file.append(i)
                c[i]="gris"
    def traite(u):
        traite_fils(u,l_adj[u])
        c[u]="noir"

    while file != []:
        traite(file.pop(0))
    return(p)



##### Algorithme final #####

def orientation(X,Y,M):
    p=produit_vec(X,Y,M)
    if p>0: return(1)
    elif p==0: return(0)
    else: return(-1)

def croisement(X,Y,P1,P2):  # Renvoie True si il y a croisement entre (XY) et (P1P2)
    if X==P1 or X==P2 or X==P1 or X==P1 or orientation(P1,P2,X)==0 or orientation(P1,P2,Y)==0:
        return False
    if orientation(P1,X,P2)==orientation(P1,X,Y) and orientation(P1,Y,X)==orientation(P1,Y,P2) and orientation(X,Y,P1) != orientation(X,Y,P2):
        return True
    else:
        return False

def croisement_polygone(X,Y,poly): # Renvoie True si [XY] ne passe pas dans le polygone (donc si il y a un problème)
    for k in range(len(poly)):
        if croisement(X,Y,poly[k-1],poly[k]):
            return True
    return False

def distance_2points(P1,P2):
    return(np.round(sqrt((P2[0]-P1[0])**2 + (P2[1]-P1[1])**2),2))

def distance_chemin(l):
    d=0
    n=len(l)
    for i in range(n-1):
        d+=distance_2points(l[i],l[i+1])
    return(np.round(d,2))

def sommets_communs(T1,T2):
    res=[]
    for x in T1:
        if x in T2:
            res.append(x)
    return(res)

print(sommets_communs([[0.3, 0], [0, 0], [0.5, -1]],[[0.3, 0], [0.5, 1], [0, 0]]))

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


def PCC_Triangulation(P,depart,arrivee):
    l_T=trianguler(P)
    D,A=trouve_depart_arrivee(l_T,depart,arrivee)
    l_adj=liste_adj(l_T)
    p=parcours_en_largeur(l_adj,D)

    i=A
    chemin=[centre_triangle(l_T[i])]
    while i != D:
        i=p[i]
        T=l_T[i]
        x=centre_triangle(T)
        chemin = [x] + chemin
    chemin=[depart]+chemin+[arrivee]
    return(chemin)

chemin=PCC_Triangulation(poly,depart,arrivee)
print(chemin)


## Résolution du problème sur PCC_Triangulation ##

def PCC_Triangulation2(P,depart,arrivee):
    l_T=trianguler(P)
    D,A=trouve_depart_arrivee(l_T,depart,arrivee)
    l_adj=liste_adj(l_T)
    p=parcours_en_largeur(l_adj,D)

    i=A
    j=A
    chemin=[centre_triangle(l_T[i])]
    while i != D:
        j=i
        i=p[i]
        T=l_T[i]
        x=centre_triangle(T)
        if len(chemin)>0 and croisement_polygone(x,chemin[0],P):
            # Cette partie sert uniquement à gérer les tracés qui passent hors limites du polygone
            S_com = sommets_communs(l_T[j],l_T[i])
            d1=distance_2points(S_com[0],x)
            d2=distance_2points(S_com[1],x)
            if d1<=d2:
                chemin = [S_com[0]] + chemin
            else:
                chemin = [S_com[1]] + chemin
        else:
            chemin = [x] + chemin

    chemin=[depart]+chemin+[arrivee]
    return(chemin)


##### Tracé graphique #####


# f = plt.figure()
#
#
# f.add_subplot(1,2,1)
# #
# # plt.plot(np.array(poly+[poly[0]])[:,0],np.array(poly+[poly[0]])[:,1],color='black')
# # plt.plot(depart[0],depart[1],marker='o',c='g',markersize=10)
# # plt.plot(arrivee[0],arrivee[1],marker='o',c='orange',markersize=10)
#
# l_triangles = trianguler(poly)
# for T in l_triangles:
#     plt.plot(np.array(T+[T[0]])[:,0],np.array(T+[T[0]])[:,1],color='blue')
# plt.plot(np.array(poly+[poly[0]])[:,0],np.array(poly+[poly[0]])[:,1],color='black')
#
#
# chemin=PCC_Triangulation(poly,depart,arrivee)
#
# plt.plot(np.array(chemin)[:,0],np.array(chemin)[:,1],color='red')
# plt.plot(depart[0],depart[1],marker='o',c='g',markersize=10)
# plt.plot(arrivee[0],arrivee[1],marker='o',c='orange',markersize=10)
#
#
#
# f.add_subplot(1,2,2)

l_triangles = trianguler(poly)
for T in l_triangles:
    plt.plot(np.array(T+[T[0]])[:,0],np.array(T+[T[0]])[:,1],color='blue')
plt.plot(np.array(poly+[poly[0]])[:,0],np.array(poly+[poly[0]])[:,1],color='black')


#chemin=PCC_Triangulation(poly,depart,arrivee)
chemin=PCC_Triangulation2(poly,depart,arrivee)

distance=distance_chemin(chemin)

plt.plot(np.array(chemin)[:,0],np.array(chemin)[:,1],color='red',linewidth=4)
plt.plot(depart[0],depart[1],marker='o',c='g',markersize=10)
plt.plot(arrivee[0],arrivee[1],marker='o',c='orange',markersize=10)



print("Distance du parcours :",distance)

agrandir1()
plt.show()

## Temps réel ##

t1=time.time()
chemin=PCC_Triangulation2(poly,depart,arrivee)
t2=time.time()

print('Temps de calcul :',t2-t1)













##### Opimisation avec choix de direction de triangulation #####

def distance_2points(P1,P2):
    return(np.round(np.sqrt((P2[0]-P1[0])**2 + (P2[1]-P1[1])**2),2))

def distance_triangle(l):
    d=0
    for T in l:
        for i in range(3):
            d+=distance_2points(T[i-1],T[i])
    return(np.round(d,2))

def distance_chemin(l):
    d=0
    n=len(l)
    for i in range(n-1):
        d+=distance_2points(l[i],l[i+1])
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
        d=distance_triangle(triangulation)
        i=2
        choix='gauche'
        for direction in ['droite','bas','haut']:
            l_t=trianguler_rec(polygone,[],direction)
            dd=distance_triangle(l_t)
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
    j=A
    chemin=[centre_triangle(l_T[i])]
    while i != D:
        j=i
        i=p[i]
        T=l_T[i]
        x=centre_triangle(T)
        if len(chemin)>0 and croisement_polygone(x,chemin[0],P):
            # Cette partie sert uniquement à gérer les tracés qui passent hors limites du polygon
            S_com = sommets_communs(l_T[j],l_T[i])
            d1=distance_2points(S_com[0],x)
            d2=distance_2points(S_com[1],x)
            if d1<=d2:
                chemin = [S_com[0]] + chemin
            else:
                chemin = [S_com[1]] + chemin
        else:
            chemin = [x] + chemin

    chemin=[depart]+chemin+[arrivee]
    return(chemin)



##### Tracé graphique "trianguler" optimisée #####

f = plt.figure()

f.add_subplot(1,2,1)

plt.plot(np.array(poly+[poly[0]])[:,0],np.array(poly+[poly[0]])[:,1],color='black')
plt.plot(depart[0],depart[1],marker='o',c='g',markersize=10)
plt.plot(arrivee[0],arrivee[1],marker='o',c='orange',markersize=10)


f.add_subplot(1,2,2)

l_triangles,choix = trianguler_opti(poly)
for T in l_triangles:
    plt.plot(np.array(T+[T[0]])[:,0],np.array(T+[T[0]])[:,1],color='blue')
plt.plot(np.array(poly+[poly[0]])[:,0],np.array(poly+[poly[0]])[:,1],color='black')


chemin=PCC_Triangulation_opti(poly,depart,arrivee)
distance=distance_chemin(chemin)

plt.plot(np.array(chemin)[:,0],np.array(chemin)[:,1],color='red')
plt.plot(depart[0],depart[1],marker='o',c='g',markersize=10)
plt.plot(arrivee[0],arrivee[1],marker='o',c='orange',markersize=10)



print('Choix de la direction de parcours :',choix)
print("Distance du parcours :",distance)

agrandir2()
plt.show()


## Temps réel #

t1=time.time()
chemin=PCC_Triangulation_opti(poly,depart,arrivee)
t2=time.time()

print('Temps de calcul réel :',t2-t1)


















##### DIJKSTRA #####

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


print(croisement(poly[2],poly[7],poly[5],poly[6]))

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
    g=(-1)*np.ones((n,n),float)
    for i in range(n):
        for j in range(i+1,n):
            if not(croisement_polygone(S[i],S[j],poly)) and dans_polygone(i,j,poly):
                g[i,j]=distance_2points(S[i],S[j])
                g[j,i]=g[i,j]
    return(g)

G=graphe(poly,depart,arrivee)
print(G)


def successeurs(s,G):
    l=[]
    for i in range(np.shape(G)[0]):
        if G[s,i]!= -1 :
            l.append(i)
    return(l)

# def ajouter_file(file,sommet,distance):
#     n=len(file)
#     bool=True
#     i=0
#     while i<n :
#         if bool and distance<file[i][1]:
#             bool=False
#             file=file[:i]+[(sommet,distance)]+file[i:]
#             i+=1
#             n+=1
#         elif sommet==file[i][0]:
#             if bool and distance>=file[i][1]:
#                 bool=False
#             else:
#                 file.pop(i)
#                 i-=1
#                 n-=1
#         i+=1
#     if bool: return(file+[(sommet,distance)])
#     else: return(file)


def ajouter_file(file,sommet,distance):
    b=len(file)-1
    a=0
    while a<=b:
        m=(a+b)//2
        if sommet==file[m][0]:
            if distance<file[m][1]:
                file.pop(m)
                b=m-2
            else:
                return(file)
        elif distance<file[m][1]:
            b=m-1
        else:
            a=m+1
    file=file[:a]+[(sommet,distance)]+file[a:]
    return(file)



def Dijkstra(G):
    n=np.shape(G)[0]

    # Position du départ : n-2  / Position de l'arrivee : n-1
    i_d,i_a=n-2,n-1

    file=[(i_d,-1)] # La valeur -1 est arbitraire, elle n'intervient jamais
    C=["blanc" for _ in range(n)]
    D=[np.infty for _ in range(n)]
    Peres=[None for _ in range(n)]

    D[i_d]=0
    Peres[i_d]= -1

    while file != []:
        pivot=file.pop(0)[0]
        C[pivot]="noir"
        if pivot==i_a:
            break
        for s in successeurs(pivot,G):
            newD=np.round(D[pivot]+G[pivot,s],2)
            if C[s]=="blanc" and (newD < D[s]):
                D[s]=newD
                Peres[s]=pivot
                file=ajouter_file(file,s,newD)
    return(Peres)

print(Dijkstra(G))

def PCC_Dijkstra(poly,depart,arrivee):
    n=len(poly)
    G=graphe(poly,depart,arrivee)

    Peres=Dijkstra(G)
    chemin=[arrivee]
    i=Peres[n+1]
    while i != n:
        chemin=[poly[i]]+chemin
        i=Peres[i]
    chemin=[depart]+chemin
    return(chemin)

print(PCC_Dijkstra(poly,depart,arrivee))


##### Tracé grahique Dijkstra #####

f = plt.figure()


f.add_subplot(1,2,1)

plt.plot(np.array(poly+[poly[0]])[:,0],np.array(poly+[poly[0]])[:,1],color='black')
plt.plot(depart[0],depart[1],marker='o',c='g',markersize=10)
plt.plot(arrivee[0],arrivee[1],marker='o',c='orange',markersize=10)



f.add_subplot(1,2,2)

plt.plot(np.array(poly+[poly[0]])[:,0],np.array(poly+[poly[0]])[:,1],color='black')

chemin=PCC_Dijkstra(poly,depart,arrivee)

plt.plot(np.array(chemin)[:,0],np.array(chemin)[:,1],color='red')
plt.plot(depart[0],depart[1],marker='o',c='g',markersize=10)
plt.plot(arrivee[0],arrivee[1],marker='o',c='orange',markersize=10)


print("Distance du parcours :",distance_chemin(chemin))

agrandir2()
plt.show()


## Temps réel ##

t1=time.time()
chemin=PCC_Dijkstra(poly,depart,arrivee)
t2=time.time()

print('Temps de calcul réel :',t2-t1)


















##### Greedy Best-First Search #####


def list_adj(poly,depart,arrivee):
    S=poly+[depart]+[arrivee]
    n=len(S)
    l=[[] for _ in range(n)]
    for i in range(n):
        for j in range(i+1,n):
            if not(croisement_polygone(S[i],S[j],poly)) and dans_polygone(i,j,poly):
                l[i].append(j)
                l[j].append(i)
    return(l)


def GBFS(poly,depart,arrivee):
    l=list_adj(poly,depart,arrivee)
    n=len(l)
    # Position du départ : n-2  / Position de l'arrivee : n-1
    i_d,i_a=n-2,n-1

    P=poly+[depart]+[arrivee]
    file=[(i_d,-1)] # la valeur -1 est arbitraire, elle n'intervient jamais
    Peres=[None for _ in range(n)]
    C=["blanc" for _ in range(n)]
    Peres[i_d]= -1

    while file != []:
        pivot=file.pop(0)[0]
        C[pivot]="noir"

        if pivot==i_a:
            break
        for s in l[pivot]:
            if C[s]=="blanc":
                Peres[s]=pivot
                d=distance_2points(P[s],arrivee)
                file=ajouter_file(file,s,d)
                C[s]="gris"
    return(Peres)


p=GBFS(poly,depart,arrivee)
print(p)


def PCC_GBFS(poly,depart,arrivee):
    n=len(poly)
    # Position du départ : n / Position de l'arrivee : n+1

    Peres=GBFS(poly,depart,arrivee)
    chemin=[arrivee]
    i=Peres[n+1]
    while i != n:
        chemin=[poly[i]]+chemin
        i=Peres[i]
    chemin=[depart]+chemin
    return(chemin)




##### Tracé grahique GBFS #####

f = plt.figure()


f.add_subplot(1,2,1)

plt.plot(np.array(poly+[poly[0]])[:,0],np.array(poly+[poly[0]])[:,1],color='black')
plt.plot(depart[0],depart[1],marker='o',c='g',markersize=10)
plt.plot(arrivee[0],arrivee[1],marker='o',c='orange',markersize=10)



f.add_subplot(1,2,2)

plt.plot(np.array(poly+[poly[0]])[:,0],np.array(poly+[poly[0]])[:,1],color='black')

chemin=PCC_GBFS(poly,depart,arrivee)

plt.plot(np.array(chemin)[:,0],np.array(chemin)[:,1],color='red')
plt.plot(depart[0],depart[1],marker='o',c='g',markersize=10)
plt.plot(arrivee[0],arrivee[1],marker='o',c='orange',markersize=10)

agrandir2()
plt.show()


## Temps réel ##

t1=time.time()
chemin=PCC_GBFS(poly,depart,arrivee)
t2=time.time()

print('Temps de calcul réel :',t2-t1)









#####       A*       #####


def graphe(poly,depart,arrivee):
    S=poly+[depart]+[arrivee]
    n=len(S)
    g=(-1)*np.ones((n,n),float)
    for i in range(n):
        for j in range(i+1,n):
            if not(croisement_polygone(S[i],S[j],poly)) and dans_polygone(i,j,poly):
                g[i,j]=distance_2points(S[i],S[j])
                g[j,i]=g[i,j]
    return(g)

G=graphe(poly,depart,arrivee)
print(G)

def Astar(poly,depart,arrivee):
    G=graphe(poly,depart,arrivee)
    n=np.shape(G)[0]

    # Position du départ : n-2  / Position de l'arrivee : n-1
    i_d,i_a=n-2,n-1

    P=poly+[depart]+[arrivee]
    file=[(i_d,-1)] # la valeur -1 est arbitraire, elle n'intervient jamais
    C=["blanc" for _ in range(n)]
    D=[np.infty for _ in range(n)]
    Peres=[None for _ in range(n)]

    D[i_d]=0
    Peres[i_d]= -1

    while file != []:
        pivot=file.pop(0)[0]
        C[pivot]="noir"

        if pivot==i_a:
            break
        for s in successeurs(pivot,G):
            newD=np.round(D[pivot]+G[pivot,s],2)
            if C[s]=="blanc" and (newD < D[s]):
                D[s]=newD
                Peres[s]=pivot
                d = np.round(newD + distance_2points(P[s],arrivee),2)
                file=ajouter_file(file,s,d)
    return(Peres)



def PCC_Astar(poly,depart,arrivee):
    n=len(poly)
    # Position du départ : n  / Position de l'arrivee : n+1

    Peres=Astar(poly,depart,arrivee)
    chemin=[arrivee]
    i=Peres[n+1]
    while i != n:
        chemin=[poly[i]]+chemin
        i=Peres[i]
    chemin=[depart]+chemin
    return(chemin)

print('-------------------------------')
print(PCC_Astar(poly,depart,arrivee))
print(PCC_Dijkstra(poly,depart,arrivee))





##### Tracé grahique A* #####

f = plt.figure()

f.add_subplot(1,2,1)

plt.plot(np.array(poly+[poly[0]])[:,0],np.array(poly+[poly[0]])[:,1],color='black')
plt.plot(depart[0],depart[1],marker='o',c='g',markersize=10)
plt.plot(arrivee[0],arrivee[1],marker='o',c='orange',markersize=10)



f.add_subplot(1,2,2)

plt.plot(np.array(poly+[poly[0]])[:,0],np.array(poly+[poly[0]])[:,1],color='black')

chemin=PCC_Astar(poly,depart,arrivee)

plt.plot(np.array(chemin)[:,0],np.array(chemin)[:,1],color='red')
plt.plot(depart[0],depart[1],marker='o',c='g',markersize=10)
plt.plot(arrivee[0],arrivee[1],marker='o',c='orange',markersize=10)


agrandir2()
plt.show()



## Temps réel ##

t1=time.time()
chemin=PCC_Astar(poly,depart,arrivee)
t2=time.time()

print('Temps de calcul réel :',t2-t1)






##### Graphique de tous les chemins trouvés #####

# f = plt.figure()
#
#
# f.add_subplot(1,2,1)
#
# l_triangles = trianguler(poly)
# for T in l_triangles:
#     plt.plot(np.array(T+[T[0]])[:,0],np.array(T+[T[0]])[:,1],color='blue')
# plt.plot(np.array(poly+[poly[0]])[:,0],np.array(poly+[poly[0]])[:,1],color='black')
# plt.plot(depart[0],depart[1],marker='o',markersize=8)
# plt.plot(arrivee[0],arrivee[1],marker='o',markersize=8)
#
#
# f.add_subplot(1,2,2)

t0=time.time()
chemin_triangulation=PCC_Triangulation2(poly,depart,arrivee)
t1=time.time()
chemin_GBFS=PCC_GBFS(poly,depart,arrivee)
t2=time.time()
chemin_Dijkstra=PCC_Dijkstra(poly,depart,arrivee)
t3=time.time()
chemin_Astar=PCC_Astar(poly,depart,arrivee)
t4=time.time()


plt.plot(np.array(poly+[poly[0]])[:,0],np.array(poly+[poly[0]])[:,1],color='black')

plt.plot(np.array(chemin_triangulation)[:,0],np.array(chemin_triangulation)[:,1],color='red',linewidth=3,label="Triangulation")
plt.plot(np.array(chemin_GBFS)[:,0],np.array(chemin_GBFS)[:,1],color='lime',linewidth=3,label='GBFS')
plt.plot(np.array(chemin_Dijkstra)[:,0],np.array(chemin_Dijkstra)[:,1],color='magenta',linewidth=3,label='Dijkstra')
plt.plot(np.array(chemin_Astar)[:,0],np.array(chemin_Astar)[:,1],color='blue',ls='--',linewidth=3,label='A*')
plt.plot(depart[0],depart[1],marker='o',c='g',markersize=10)
plt.plot(arrivee[0],arrivee[1],marker='o',c='orange',markersize=10)
plt.legend(loc='lower right',fontsize=23)


agrandir1()
plt.show()


d1 = distance_chemin(chemin_triangulation)
d2 = distance_chemin(chemin_GBFS)
d3 = distance_chemin(chemin_Dijkstra)
d4 = distance_chemin(chemin_Astar)

print("Distance du parcours avec Triangulation :",d1)
print('Temps de calcul réel :',np.round(t1-t0,5),'\n')
print("Distance du parcours avec Greedy Best-First Search :",d2)
print('Temps de calcul réel :',np.round(t2-t1,5),'\n')
print("Distance du parcours avec Dijkstra :",d3)
print('Temps de calcul réel :',np.round(t3-t2,5),'\n')
print("Distance du parcours avec A* :",d4)
print('Temps de calcul réel :',np.round(t4-t3,5),'\n')


##### Temps de calculs #####

## Triangulation ##

N=1000

l=[]
for _ in range(N):
    t0=time.time()
    chemin_triangulation=PCC_Triangulation2(poly,depart,arrivee)
    t1=time.time()
    chemin_triangulation=[]
    l.append(t1-t0)

print(np.round(np.mean(l),5))


## GBFS ##

N=1000

l=[]
for _ in range(N):
    t0=time.time()
    chemin_GBFS=PCC_GBFS(poly,depart,arrivee)
    t1=time.time()
    chemin_GBFS=[]
    l.append(t1-t0)

print(np.round(np.mean(l),5))


## Dijkstra ##

N=1000

l=[]
for _ in range(N):
    t0=time.time()
    chemin_Dijkstra=PCC_Dijkstra(poly,depart,arrivee)
    t1=time.time()
    l.append(t1-t0)

print(np.round(np.mean(l),5))


## A* ##

N=1000

l=[]
for _ in range(N):
    t0=time.time()
    chemin_Astar=PCC_Astar(poly,depart,arrivee)
    t1=time.time()
    l.append(t1-t0)

print(np.round(np.mean(l),5))
























