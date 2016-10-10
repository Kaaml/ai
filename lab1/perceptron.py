import numpy as np
import random
import os, subprocess
import matplotlib.pyplot as plt
 
class Perceptron:
    def __init__(self, N):
        xA,yA,xB,yB = [random.uniform(-1, 1) for i in range(4)]
        self.V = np.array([xB*yA-xA*yB, yB-yA, xA-xB])
        self.X = self.generate_points(N)
# generate linear separeble data 
    def generate_points(self, N):
        X = []
        a, b = [random.uniform(-1,1), random.uniform(-0.5, 0.5)]
        for i in range(N):
            x, y = [random.uniform(-1,1) for j in range(2)]
            fx = a*x+b
            s = -1
            if y >= fx :
              s = 1
            X.append( (np.array([1,x,y]), s ))
        #print(X)
        return X
 
    def plot(self, mispts=None, vec=None, save=False):
        fig = plt.figure(figsize=(5,5))
        plt.xlim(-1,1)
        plt.ylim(-1,1)
        V = self.V
        a, b = -V[1]/V[2], -V[0]/V[2]
        l = np.linspace(-1,1)
        plt.plot(l, a*l+b, 'k-')
        cols = {1: 'r', -1: 'b'}
        for x,s in self.X:
            plt.plot(x[1], x[2], cols[s]+'o')
        if mispts:
            for x,s in mispts:
                plt.plot(x[1], x[2], cols[s]+'.')
        if vec != None:
            aa, bb = -vec[1]/vec[2], -vec[0]/vec[2]
            plt.plot(l, aa*l+bb, 'g-', lw=2)
        plt.title('a = %s b = %s' % (str(aa),str(bb)))
        print( 'a = %s b = %s' % (str(aa), str(bb) ) )
        if save:
            if not mispts:
                plt.title('N = %s' % (str(len(self.X))))
            else:
                plt.title('N = %s with %s test points' \
                          % (str(len(self.X)),str(len(mispts))))
            plt.savefig('p_N%s' % (str(len(self.X))), \
                        dpi=200, bbox_inches='tight')
 
    def classification_error(self, vec, pts=None):
        if not pts:
            pts = self.X
        M = len(pts)
        n_mispts = 0
        for x,s in pts:
            if int(np.sign(vec.T.dot(x))) != s:
                n_mispts += 1
        error = n_mispts / float(M)
        return error
 
    def choose_miscl_point(self, vec):
        pts = self.X
        mispts = []
        for x,s in pts:
            if int(np.sign(vec.T.dot(x))) != s:
                mispts.append((x, s))
        return mispts[random.randrange(0,len(mispts))]
 
    def pla(self, save=False):
        w = np.zeros(3) #c b a
        X, N = self.X, len(self.X)
        it = 0
        while self.classification_error(w) != 0:
            it += 1
            x, s = self.choose_miscl_point(w)
            w += s*x
            if save:
                self.plot(vec=w)
                #plt.title('N = %s, Iteration %s\n' \
                #          % (str(N),str(it)))
                plt.savefig('p_N%s_it%s' % (str(N),str(it)), \
                            dpi=200, bbox_inches='tight')
        self.w = w
        print( w );
 
    def check_error(self, M, vec):
        check_pts = self.generate_points(M)
        return self.classification_error(vec, pts=check_pts)
