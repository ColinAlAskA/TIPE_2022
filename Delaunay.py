##### Triangulation de Delaunay #####

from math import *
import time
import random as r
import copy as c
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from itertools import combinations

def agrandir2():
    mngr = plt.get_current_fig_manager()
    mngr.window.setGeometry(50,50,1000,1000)
    f.set_tight_layout(True)
    #plt.tight_layout(pad=0.1)

##

points = [[r.randint(-500,500)/100,r.randint(-500,500)/100] for _ in range(50)]
#poly = [[-1.5,1.94],[-2.46,-4.3],[3.18,-1.8],[5.7,-6.84],[10,-4],[4.9,-1.56],[12,-1],[5.2,0.42],[11.86,2.94],[5,2],[4.62,4.04],[3.92,2.28],[1.42,2.48],[-0.76,3.72],[0,1.56],[3.28,0.92],[1.82,-1]]
poly = [[-3.37,15.03],[-9.44,11.54],[-4.31,11.75],[-8.45,6.59],[-9.61,2.98],[-16.96,-1.94],[-12.48,-0.72],[-6.99,-4.09],[0.86,0.86],[-2.17,6.39],[3.57,3.73],[3.07,-2.09],[0.78,-5.25],[5.94,-2.26],[7.13,4.71],[12.76,-2.59],[17.08,3.52],[11.37,2.18],[26.93,13.4],[6.48,6.39],[2.78,5.68],[-0.55,8.51],[8.6,15.7],[14.79,12.7],[16.79,14.95],[12.46,16.65],[1.2,16.4],[5.44,15.32],[2.03,15.28],[-6.84,1.19],[-6.02,4.49],[-0.71,15.24],[-2.29,16.86]]



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

        X=(C1*B2-C2*B1) / (A2*B1-A1*B2)
        Y=(C1*A2-A1*C2) / (A1*B2-B1*A2)

        R=(X-T[1][0])**2 + (Y-T[1][1])**2

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

print(Delaunay(poly))

t1=time.time()

f = plt.figure()

f.add_subplot(1,2,1)

l_triangles = trianguler(poly)
for T in l_triangles:
    plt.plot(np.array(T+[T[0]])[:,0],np.array(T+[T[0]])[:,1],color='black')

f.add_subplot(1,2,2)


l_triangles=Delaunay(poly)
for T in l_triangles:
    plt.plot(np.array(list(T)+[T[0]])[:,0],np.array(list(T)+[T[0]])[:,1],color='black')

agrandir2()
plt.show()

t2=time.time()

print('Temps de calcul :',t2-t1)




## TEST tracé d'un cercle circonscrit ##


import matplotlib.patches as patches

f,ax= plt.subplots()



l_triangles=Delaunay(poly)
for T in l_triangles:
    plt.plot(np.array(list(T)+[T[0]])[:,0],np.array(list(T)+[T[0]])[:,1],color='black')

T=([3.18, -1.8], [4.9, -1.56], [3.28, 0.92])

A1=2*(T[1][0]-T[0][0])
B1=2*(T[1][1]-T[0][1])
C1=T[0][0]**2 + T[0][1]**2 - T[1][0]**2 - T[1][1]**2
A2=2*(T[2][0]-T[1][0])
B2=2*(T[2][1]-T[1][1])
C2=T[1][0]**2 + T[1][1]**2 - T[2][0]**2 - T[2][1]**2

X=(C1*B2-C2*B1) / (A2*B1-A1*B2)
Y=(C1*A2-A1*C2) / (A1*B2-B1*A2)
R=(X-T[1][0])**2 + (Y-T[1][1])**2


Cercle = plt.Circle((X,Y),radius=sqrt(R),fc='white',ec='blue')
ax.add_patch(Cercle)

plt.show()


##

import random as r
from itertools import combinations

points = [[r.randint(-500,500),r.randint(-500,500)] for _ in range(20)]
poly = [[-1.5,1.94],[-2.46,-4.3],[3.18,-1.8],[5.7,-6.84],[10,-4],[4.9,-1.56],[12,-1],[5.2,0.42],[11.86,2.94],[5,2],[4.62,4.04],[3.92,2.28],[1.42,2.48],[-0.76,3.72],[0,1.56],[3.28,0.92],[1.82,-1]]

print(points)

def Delaunay2(ListePoints):
    ListeTriangle = list(combinations(ListePoints,3))
    Triangulation=[]
    Liste_C_R=[]
    for T in ListeTriangle:
        bool=True
        ListD=[]

        A1=2*(T[1][0]-T[0][0])
        B1=2*(T[1][1]-T[0][1])
        C1=T[0][0]**2 + T[0][1]**2 - T[1][0]**2 - T[1][1]**2

        A2=2*(T[2][0]-T[1][0])
        B2=2*(T[2][1]-T[1][1])
        C2=T[1][0]**2 + T[1][1]**2 - T[2][0]**2 - T[2][1]**2

        X=(C1*B2-C2*B1) / (A2*B1-A1*B2)
        Y=(C1*A2-A1*C2) / (A1*B2-B1*A2)

        R=(X-T[1][0])**2 + (Y-T[1][1])**2

        for P in ListePoints:
            if not(P in T):
                distance=(P[0]-X)**2+(P[1]-Y)**2
                ListD.append(distance)
                for D in ListD:
                    if D<=R:
                        bool=False
                        break
        if bool :
            C=(np.round(X,2),np.round(Y,2))
            RR=np.round(np.sqrt(R),2)
            Triangulation.append(T)
            Liste_C_R.append([C,RR])
    return(Triangulation,Liste_C_R)

f,ax= plt.subplots()

T,L=Delaunay2(poly)
n=len(L)
for i in range(n):
    plt.plot(np.array(list(T[i])+[T[i][0]])[:,0],np.array(list(T[i])+[T[i][0]])[:,1],color='black')
    ax.add_patch(plt.Circle(L[i][0],radius=L[i][1],fc='None',ec='blue'))

agrandir2()
plt.show()


## Méthode de retournement - Triangulation ##


def cercle_circonscrit(T):
    A1=2*(T[1][0]-T[0][0])
    B1=2*(T[1][1]-T[0][1])
    C1=T[0][0]**2 + T[0][1]**2 - T[1][0]**2 - T[1][1]**2

    A2=2*(T[2][0]-T[1][0])
    B2=2*(T[2][1]-T[1][1])
    C2=T[1][0]**2 + T[1][1]**2 - T[2][0]**2 - T[2][1]**2

    X=(C1*B2-C2*B1) / (A2*B1-A1*B2)
    Y=(C1*A2-A1*C2) / (A1*B2-B1*A2)

    R=(X-T[1][0])**2 + (Y-T[1][1])**2

    centre=(np.round(X,2),np.round(Y,2))
    rayon=np.round(np.sqrt(R),2)

    return(centre,rayon)

# Trop compliqué pour pas grand chose...


































####### Modules trouvés sur internet ########





from scipy.spatial import Delaunay
import time
import matplotlib.pyplot as plt
import copy as c

poly = [[0,0],[0.5,-1],[1.5,-0.2],[2,-0.5],[2,0],[1.5,1],[0.3,0],[0.5,1]]


#poly = [[-1.5,1.94],[-2.46,-4.3],[3.18,-1.8],[5.7,-6.84],[10,-4],[4.9,-1.56],[12,-1],[5.2,0.42],[11.86,2.94],[5,2],[4.62,4.04],[3.92,2.28],[1.42,2.48],[-0.76,3.72],[0,1.56],[3.28,0.92],[1.82,-1]]

#poly = [[-3.37,15.03],[-9.44,11.54],[-4.31,11.75],[-8.45,6.59],[-9.61,2.98],[-16.96,-1.94],[-12.48,-0.72],[-6.99,-4.09],[0.86,0.86],[-2.17,6.39],[3.57,3.73],[3.07,-2.09],[0.78,-5.25],[5.94,-2.26],[7.13,4.71],[12.76,-2.59],[17.08,3.52],[11.37,2.18],[26.93,13.4],[6.48,6.39],[2.78,5.68],[-0.55,8.51],[8.6,15.7],[14.79,12.7],[16.79,14.95],[12.46,16.65],[1.2,16.4],[5.44,15.32],[2.03,15.28],[-6.20,1.19],[-6.02,4.49],[-0.71,15.24],[-2.29,16.86]]

poly = np.array(poly)

t1=time.time()
tri = Delaunay(poly)
t2=time.time()

print('Temps nécessaire :',t2-t1)

plt.triplot(poly[:,0], poly[:,1], tri.simplices,color='black')
plt.plot(poly[:,0], poly[:,1], 'o')
plt.show()

##

def transforme(l):
    if l==[]:
        return(l)
    else:
        [x,y]=l[0]
        return([(x,y)]+transforme(l[1:]))
##

#poly = [[0,0],[0.5,-1],[1.5,-0.2],[2,-0.5],[2,0],[1.5,1],[0.3,0],[0.5,1]]

#poly = [[-1.5,1.94],[-2.46,-4.3],[3.18,-1.8],[5.7,-6.84],[10,-4],[4.9,-1.56],[12,-1],[5.2,0.42],[11.86,2.94],[5,2],[4.62,4.04],[3.92,2.28],[1.42,2.48],[-0.76,3.72],[0,1.56],[3.28,0.92],[1.82,-1]]

poly = [[-3.37,15.03],[-9.44,11.54],[-4.31,11.75],[-8.45,6.59],[-9.61,2.98],[-16.96,-1.94],[-12.48,-0.72],[-6.99,-4.09],[0.86,0.86],[-2.17,6.39],[3.57,3.73],[3.07,-2.09],[0.78,-5.25],[5.94,-2.26],[7.13,4.71],[12.76,-2.59],[17.08,3.52],[11.37,2.18],[26.93,13.4],[6.48,6.39],[2.78,5.68],[-0.55,8.51],[8.6,15.7],[14.79,12.7],[16.79,14.95],[12.46,16.65],[1.2,16.4],[5.44,15.32],[2.03,15.28],[-6.20,1.19],[-6.02,4.49],[-0.71,15.24],[-2.29,16.86]]

poly= transforme(poly)

import triangle as tr

init=dict (vertices= poly)

mesh = tr.triangulate(init)

Figure = plt.figure()
Axes = Figure.add_subplot(111)
for tri in mesh["triangles"] :
    for j in range(len(tri)):
        XA=mesh["vertices"][tri[j-1]][0] #Coordonnée X de la première extrémité
        XB=mesh["vertices"][tri[j]][0]   #Coordonnée X de la seconde extrémité
        YA=mesh["vertices"][tri[j-1]][1] #Coordonnée Y de la première extrémité
        YB=mesh["vertices"][tri[j]][1]   #Coordonnée Y de la seconde extrémité
        Axes.plot([XA,XB],[YA,YB],c="black")
plt.show()

##

# -*- coding: ascii -*-
"""
Simple structured Delaunay triangulation in 2D with Bowyer-Watson algorithm.

Written by Jose M. Espadero ( http://github.com/jmespadero/pyDelaunay2D )
Based on code from Ayron Catteau. Published at http://github.com/ayron/delaunay

Just pretend to be simple and didactic. The only requisite is numpy.
Robust checks disabled by default. May not work in degenerate set of points.
"""

import numpy as np
from math import sqrt


class Delaunay2D:
    """
    Class to compute a Delaunay triangulation in 2D
    ref: http://en.wikipedia.org/wiki/Bowyer-Watson_algorithm
    ref: http://www.geom.uiuc.edu/~samuelp/del_project.html
    """

    def __init__(self, center=(0, 0), radius=9999):
        """ Init and create a new frame to contain the triangulation
        center -- Optional position for the center of the frame. Default (0,0)
        radius -- Optional distance from corners to the center.
        """
        center = np.asarray(center)
        # Create coordinates for the corners of the frame
        self.coords = [center+radius*np.array((-1, -1)),
                       center+radius*np.array((+1, -1)),
                       center+radius*np.array((+1, +1)),
                       center+radius*np.array((-1, +1))]

        # Create two dicts to store triangle neighbours and circumcircles.
        self.triangles = {}
        self.circles = {}

        # Create two CCW triangles for the frame
        T1 = (0, 1, 3)
        T2 = (2, 3, 1)
        self.triangles[T1] = [T2, None, None]
        self.triangles[T2] = [T1, None, None]

        # Compute circumcenters and circumradius for each triangle
        for t in self.triangles:
            self.circles[t] = self.circumcenter(t)

    def circumcenter(self, tri):
        """Compute circumcenter and circumradius of a triangle in 2D.
        Uses an extension of the method described here:
        http://www.ics.uci.edu/~eppstein/junkyard/circumcenter.html
        """
        pts = np.asarray([self.coords[v] for v in tri])
        pts2 = np.dot(pts, pts.T)
        A = np.bmat([[2 * pts2, [[1],
                                 [1],
                                 [1]]],
                      [[[1, 1, 1, 0]]]])

        b = np.hstack((np.sum(pts * pts, axis=1), [1]))
        x = np.linalg.solve(A, b)
        bary_coords = x[:-1]
        center = np.dot(bary_coords, pts)

        # radius = np.linalg.norm(pts[0] - center) # euclidean distance
        radius = np.sum(np.square(pts[0] - center))  # squared distance
        return (center, radius)

    def inCircleFast(self, tri, p):
        """Check if point p is inside of precomputed circumcircle of tri.
        """
        center, radius = self.circles[tri]
        return np.sum(np.square(center - p)) <= radius

    def inCircleRobust(self, tri, p):
        """Check if point p is inside of circumcircle around the triangle tri.
        This is a robust predicate, slower than compare distance to centers
        ref: http://www.cs.cmu.edu/~quake/robust.html
        """
        m1 = np.asarray([self.coords[v] - p for v in tri])
        m2 = np.sum(np.square(m1), axis=1).reshape((3, 1))
        m = np.hstack((m1, m2))    # The 3x3 matrix to check
        return np.linalg.det(m) <= 0

    def addPoint(self, p):
        """Add a point to the current DT, and refine it using Bowyer-Watson.
        """
        p = np.asarray(p)
        idx = len(self.coords)
        # print("coords[", idx,"] ->",p)
        self.coords.append(p)

        # Search the triangle(s) whose circumcircle contains p
        bad_triangles = []
        for T in self.triangles:
            # Choose one method: inCircleRobust(T, p) or inCircleFast(T, p)
            if self.inCircleFast(T, p):
                bad_triangles.append(T)

        # Find the CCW boundary (star shape) of the bad triangles,
        # expressed as a list of edges (point pairs) and the opposite
        # triangle to each edge.
        boundary = []
        # Choose a "random" triangle and edge
        T = bad_triangles[0]
        edge = 0
        # get the opposite triangle of this edge
        while True:
            # Check if edge of triangle T is on the boundary...
            # if opposite triangle of this edge is external to the list
            tri_op = self.triangles[T][edge]
            if tri_op not in bad_triangles:
                # Insert edge and external triangle into boundary list
                boundary.append((T[(edge+1) % 3], T[(edge-1) % 3], tri_op))

                # Move to next CCW edge in this triangle
                edge = (edge + 1) % 3

                # Check if boundary is a closed loop
                if boundary[0][0] == boundary[-1][1]:
                    break
            else:
                # Move to next CCW edge in opposite triangle
                edge = (self.triangles[tri_op].index(T) + 1) % 3
                T = tri_op

        # Remove triangles too near of point p of our solution
        for T in bad_triangles:
            del self.triangles[T]
            del self.circles[T]

        # Retriangle the hole left by bad_triangles
        new_triangles = []
        for (e0, e1, tri_op) in boundary:
            # Create a new triangle using point p and edge extremes
            T = (idx, e0, e1)

            # Store circumcenter and circumradius of the triangle
            self.circles[T] = self.circumcenter(T)

            # Set opposite triangle of the edge as neighbour of T
            self.triangles[T] = [tri_op, None, None]

            # Try to set T as neighbour of the opposite triangle
            if tri_op:
                # search the neighbour of tri_op that use edge (e1, e0)
                for i, neigh in enumerate(self.triangles[tri_op]):
                    if neigh:
                        if e1 in neigh and e0 in neigh:
                            # change link to use our new triangle
                            self.triangles[tri_op][i] = T

            # Add triangle to a temporal list
            new_triangles.append(T)

        # Link the new triangles each another
        N = len(new_triangles)
        for i, T in enumerate(new_triangles):
            self.triangles[T][1] = new_triangles[(i+1) % N]   # next
            self.triangles[T][2] = new_triangles[(i-1) % N]   # previous

    def exportTriangles(self):
        """Export the current list of Delaunay triangles
        """
        # Filter out triangles with any vertex in the extended BBox
        return [(a-4, b-4, c-4)
                for (a, b, c) in self.triangles if a > 3 and b > 3 and c > 3]

    def exportCircles(self):
        """Export the circumcircles as a list of (center, radius)
        """
        # Remember to compute circumcircles if not done before
        # for t in self.triangles:
        #     self.circles[t] = self.circumcenter(t)

        # Filter out triangles with any vertex in the extended BBox
        # Do sqrt of radius before of return
        return [(self.circles[(a, b, c)][0], sqrt(self.circles[(a, b, c)][1]))
                for (a, b, c) in self.triangles if a > 3 and b > 3 and c > 3]

    def exportDT(self):
        """Export the current set of Delaunay coordinates and triangles.
        """
        # Filter out coordinates in the extended BBox
        coord = self.coords[4:]

        # Filter out triangles with any vertex in the extended BBox
        tris = [(a-4, b-4, c-4)
                for (a, b, c) in self.triangles if a > 3 and b > 3 and c > 3]
        return coord, tris

    def exportExtendedDT(self):
        """Export the Extended Delaunay Triangulation (with the frame vertex).
        """
        return self.coords, list(self.triangles)

    def exportVoronoiRegions(self):
        """Export coordinates and regions of Voronoi diagram as indexed data.
        """
        # Remember to compute circumcircles if not done before
        # for t in self.triangles:
        #     self.circles[t] = self.circumcenter(t)
        useVertex = {i: [] for i in range(len(self.coords))}
        vor_coors = []
        index = {}
        # Build a list of coordinates and one index per triangle/region
        for tidx, (a, b, c) in enumerate(sorted(self.triangles)):
            vor_coors.append(self.circles[(a, b, c)][0])
            # Insert triangle, rotating it so the key is the "last" vertex
            useVertex[a] += [(b, c, a)]
            useVertex[b] += [(c, a, b)]
            useVertex[c] += [(a, b, c)]
            # Set tidx as the index to use with this triangle
            index[(a, b, c)] = tidx
            index[(c, a, b)] = tidx
            index[(b, c, a)] = tidx

        # init regions per coordinate dictionary
        regions = {}
        # Sort each region in a coherent order, and substitude each triangle
        # by its index
        for i in range(4, len(self.coords)):
            v = useVertex[i][0][0]  # Get a vertex of a triangle
            r = []
            for _ in range(len(useVertex[i])):
                # Search the triangle beginning with vertex v
                t = [t for t in useVertex[i] if t[0] == v][0]
                r.append(index[t])  # Add the index of this triangle to region
                v = t[1]            # Choose the next vertex to search
            regions[i-4] = r        # Store region.

        return vor_coors, regions

##


#!/usr/bin/env python3
"""
Minimal delaunay2D test
See: http://github.com/jmespadero/pyDelaunay2D
"""
import numpy as np

# Create a random set of 2D points
seeds = np.random.random((10, 2))

# Create Delaunay Triangulation and insert points one by one
dt = Delaunay2D()
for s in seeds:
    dt.addPoint(s)

# Dump points and triangles to console
print("Input points:\n", seeds)
print ("Delaunay triangles:\n", dt.exportTriangles())





##

#!/usr/bin/env python3
# -*- coding: ascii -*-
"""
Simple delaunay2D demo with mathplotlib
Written by Jose M. Espadero < http://github.com/jmespadero/pyDelaunay2D >
"""
import numpy as np

if __name__ == '__main__':

    # Generate 'numSeeds' random seeds in a square of size 'radius'
    numSeeds = 20
    radius = 100
    seeds = radius * np.random.random((numSeeds, 2))
    print("seeds:\n", seeds)
    print("BBox Min:", np.amin(seeds, axis=0),
          "Bbox Max: ", np.amax(seeds, axis=0))

    """
    Compute our Delaunay triangulation of seeds.
    """
    # It is recommended to build a frame taylored for our data
    # dt = D.Delaunay2D() # Default frame
    center = np.mean(seeds, axis=0)
    dt = Delaunay2D(center, 50 * radius)

    # Insert all seeds one by one
    for s in seeds:
        dt.addPoint(s)

    # Dump number of DT triangles
    print (len(dt.exportTriangles()), "Delaunay triangles")

    """
    Demostration of how to plot the data.
    """
    import matplotlib.pyplot as plt
    import matplotlib.tri
    import matplotlib.collections

    # Create a plot with matplotlib.pyplot
    fig, ax = plt.subplots()
    ax.margins(0.1)
    ax.set_aspect('equal')
    plt.axis([-1, radius+1, -1, radius+1])

    # Plot our Delaunay triangulation (plot in blue)
    cx, cy = zip(*seeds)
    dt_tris = dt.exportTriangles()
    ax.triplot(matplotlib.tri.Triangulation(cx, cy, dt_tris), 'bo--')

    # Plot annotated Delaunay vertex (seeds)
    """
    for i, v in enumerate(seeds):
        plt.annotate(i, xy=v)
    """

    # DEBUG: Use matplotlib to create a Delaunay triangulation (plot in green)
    # DEBUG: It should be equal to our result in dt_tris (plot in blue)
    # DEBUG: If boundary is diferent, try to increase the value of your margin
    # ax.triplot(matplotlib.tri.Triangulation(*zip(*seeds)), 'g--')

    # DEBUG: plot the extended triangulation (plot in red)
    # edt_coords, edt_tris = dt.exportExtendedDT()
    # edt_x, edt_y = zip(*edt_coords)
    # ax.triplot(matplotlib.tri.Triangulation(edt_x, edt_y, edt_tris), 'ro-.')

    # Plot the circumcircles (circles in black)
    """
    for c, r in dt.exportCircles():
        ax.add_artist(plt.Circle(c, r, color='k', fill=False, ls='dotted'))
    """

    # Build Voronoi diagram as a list of coordinates and regions
    vc, vr = dt.exportVoronoiRegions()

    # Plot annotated voronoi vertex
    """
    plt.scatter([v[0] for v in vc], [v[1] for v in vc], marker='.')
    for i, v in enumerate(vc):
        plt.annotate(i, xy=v)
    """

    # Plot annotated voronoi regions as filled polygons
    """
    for r in vr:
        polygon = [vc[i] for i in vr[r]]     # Build polygon for each region
        plt.fill(*zip(*polygon), alpha=0.2)  # Plot filled polygon
        plt.annotate("r%d" % r, xy=np.average(polygon, axis=0))
    """

    # Plot voronoi diagram edges (in red)
    for r in vr:
        polygon = [vc[i] for i in vr[r]]       # Build polygon for each region
        plt.plot(*zip(*polygon), color="red")  # Plot polygon edges in red

    # Dump plot to file
    # plt.savefig('output-delaunay2D.png', dpi=96)
    # plt.savefig('output-delaunay2D.svg', dpi=96)

    plt.show()

    # Plot a step-by-step triangulation
    '''
    #Starts from a new Delaunay2D frame
    dt2 = Delaunay2D(center, 50 * radius)
    for i,s in enumerate(seeds):
        print("Inserting seed", i, s)
        dt2.addPoint(s)
        if i > 1:
            fig, ax = plt.subplots()
            ax.margins(0.1)
            ax.set_aspect('equal')
            plt.axis([-1, radius+1, -1, radius+1])
            for i, v in enumerate(seeds):
                plt.annotate(i, xy=v)              # Plot all seeds
            for t in dt2.exportTriangles():
                polygon = [seeds[i] for i in t]     # Build polygon for each region
                plt.fill(*zip(*polygon), fill=False, color="b")  # Plot filled polygon
            plt.show()
    '''















