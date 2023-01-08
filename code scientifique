import numpy as np
import math
import random
import scipy.stats
import matplotlib.pyplot as plt


def samples(a, b, n):
    w = [1/n] * n
    x = np.random.normal(a, b, n).tolist()
    return w, x

def gausienne(x,mu,sigma):
    return 1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(- (x - mu)**2 / (2 * sigma**2))

def expon(x,l):
  if np.all(x < 0) :
    return 0
  else :
    return l*np.exp(-l*x)

def posterior(D,x,C,i):
  if i ==0 : 
    return prior(D,x,C,0)
  else :
    return prior(D,x,C,i) * likelihood(D[-1], x, C[-1])
    
def mean(w,x:list):
    u=0
    for i in range(len(w)):
        u+=w[i]*x[i]
    return u 

def cov(w,x):
    u=mean(w,x)
    c=0
    for i in range(len(w)):
        c+=w[i]*x[i]**2-u**2
    return c   

def mean_f(w,x,f):
    u=0
    for i in range(len(w)):
        u=u+w[i]*f(x[i])
    return u  

def update(W, X, D, C,j):
    # Calculate the updated weights
    updated_weights = []
    for i in range(len(X)):
        updated_weights.append(W[i] * posterior(D,X[i],C,j))
    # Normalize the weights
    if 0 in updated_weights:
    # Replace 0 with a very small value
        updated_weights = [w if w > 0 else 1e-10 for w in updated_weights]
    updated_weights = [w / np.sum(updated_weights) for w in updated_weights]
    
    return updated_weights

def likelihood(D:float, x:float, C:float):
    t = C
    if np.all(abs(D-1)<abs(D)):
        return np.exp(-t/100) * ((np.cos(x*t/2))**2) + (1 - np.exp(-t/100)) / 2  
    else :
        return 1-likelihood(1, x, C)

def prior(D,x,C,i):
    if i == 0:
        return expon(x, 0.5)
    else : 
        return posterior(D, x, C,i-1)
    
def resample(w:list, x, a):
    # Calculate the mean (mu)
    mu = mean(w, x)
    
    # Calculate the covariance (Sigma)
    s = cov(w, x)
    
    # Calculate the unnormalized updated weights
    h = np.sqrt(1 - a**2)
    ai = abs(h**2 * s)
    for i in range(len(w)):
        j = random.randrange(0, len(w), 1)
        u=a*x[j]+(1-a)*mu
        x.append(np.random.normal(u,ai,len(x))[i])
        w =[1/len(x) for _ in range(len(x))]
    return w, x

def utilnv(D,w,x,d,c,j):
    ud=0
    w1=[]
    so=0
    a=0
    for e in d:
        D+=[e]
        w1=update(w,x,D,c)
        u=mean(w,x)
        for i in range(len(x)):
            ud=ud+w[i]*(x[i]-u)**2
        for i in range(len(x)):
            so=so+w[i]*posterior(D,x[i],c,j)
        ud=ud*so
        a+=ud
        ud=0
        D.remove(e)
    return  a

def utillg(w,x,c,d,D,j):
    pd=[0]*len(x)
    pd1=0
    s=0
    f=0
    H=0
    for e in d:
        D+=[e]
        for i in range(len(x)):
            pd[i]=posterior(D,x[i],c,j)
        for i in range(len(w)):
            pd1=pd1+w[i]*pd[i]
        for i in range(len(x)):
            if pd[i]!=0:
                H+=pd[i]*np.log(1/pd[i])
        for i in range(len(x)):
            f= w[i]*H
        if pd1!=0:
          s+= pd1*np.log(1/pd1)-f
        pd1=0
        D.remove(e)
    return s

def get_si(w):
    s = []
    while w!=[]:
        a = max(w)
        i = 0
        while i < len(w):
            if w[i] == a:
                s.append(i)
                w.pop(i)
            else:
                i += 1
    return s

def diff(f1,f2,D1,D2,C1,C2,i):
    a=0
    d1=0
    d2=0
# intervalle de comparaison
    x_min, x_max = 0, 1

# nombre de points d'évaluation
    n_points = 100

# génère une grille de points d'évaluation
    x = np.linspace(x_min, x_max, n_points)

# évalue les deux fonctions sur la grille
    y1 = f1(D1,x,C1,i)
    y2 = f2(D2,x,C2,i)

# calcule la différence absolue entre les deux fonctions
    diff = y1 - y2
    for i in range(len(diff)):
        a+=diff[i]
        d1+=y1[i]
        d2+=y2[i]
    if d1 > d2:
      A=[D1]*int (d1/d2)+[D2]    
    else:
      A=[D2]*int (d2/d1)+[D1]
    random.shuffle(A)
    k=A[0]
    return k
    
    
# Test the function with a sample list
def reapprox(w,x, approx_ratio):
    n=int(len(w)*approx_ratio)
    elements = list(range(len(w)))

# Shuffle the list of elements
    np.random.shuffle(elements)

# The shuffled list is the permutation pi
    pi = elements
    wp=[]
    xp=[]
    for i in range(len(w)):
        wp.append(w[pi[i]])
        xp.append(x[pi[i]])
    s=get_si(wp)
    wn=[]
    xn=[]
    for i in range(len(s)):
        wn.append(w[s[i]])
        xn.append(x[s[i]])
    return wn,xn    
import random

def choose_experiments(D, w, x, c, d, utility_function,j):
    # Définir les bornes de la recherche dichotomique
    lower_bound = 0
    upper_bound = 100
    c1=c
    # Trouver l'utilité maximale en utilisant une recherche dichotomique
    while upper_bound - lower_bound > 0.01:
        t = (upper_bound + lower_bound) / 2
        c+=[t]
        c1+=[t+0.01]
        utility = utility_function(w, x, c, d, D,j)
        if utility > utility_function(w, x, c1, d, D,j):
            upper_bound = t
        else:
            lower_bound = t
    
    # Renvoyer le paramètre de contrôle qui maximise l'utilité
    return t


# Test the function

def LocalOptimize(Util,D,C,w, x,j):
    d=[0,1]
    return C, utillg(w, x, C, d, D,j)


#algo7
def EstimateAdaptive(n, mu, sigma, N, a, resample_threshold, approx_ratio, Util, n_guesses):
    # Initialize the weights and locations to be uniformly distributed
    weights = np.ones(n) / n
    locations = samples(mu, sigma,n)[1]
    C_tot=[]
    D_tot=[]
    D=[0,1]
    mu, sigma = mu, sigma
    max=0
    # Loop over the number of experiments
    for i in range(1, N+1):
        # Reapproximate the distribution if necessary
        if approx_ratio != 1:
            w, x = reapprox(weights, locations, approx_ratio)
        else:
            w, x = weights, locations
        for iexp in range(1,n_guesses+1):
        # Make guesses for the control of the experiment
            C_guesses = C_tot + [choose_experiments(D_tot, w, x, C_tot, D, Util,i)]
            C_guesses.sort()
        # Optimize the utility for each guess
            C_hat_guesses = []
            U_guesses = []
            C_hat, U = LocalOptimize(Util, D_tot, C_guesses, w, x,i)
            C_hat_guesses.append(C_hat)
            U_guesses.append(U)
        
        # Choose the guess that maximizes the utility
        i_best = np.argmax(U_guesses)
        C_hat = C_hat_guesses[i_best]
        C_tot = C_hat
        # Perform the experiment and update the weights and locations
        D0=D_tot +[D[0]]
        D1=D_tot +[D[1]]
        A = diff(posterior,posterior,D0,D1,C_hat,C_hat,i)
        D_tot = A
        weights = update(weights, locations, D_tot, C_hat,i)
        weights, locations = reapprox(weights, locations,approx_ratio)
        weights = [w / np.sum(weights) for w in weights]

        s=0
        for i in range(len(weights)):
            s+= weights[i]**2
        # Resample if necessary
        if 1/s < resample_threshold:
            weights, locations = resample(weights, locations, a)
    print(D_tot)
    # Return the mean as an estimate of the true model
    X = np.linspace(-10, 10, 100) 
    y = [posterior(D_tot,x,C_tot,N) for x in X]

    # affiche le graphique
    plt.plot(X,y)
    plt.scatter(locations, [0] * len(locations), c='red')
    plt.scatter(mean(weights, locations), [0] , c='green')
    plt.show()
    return mean(weights, locations)
print(EstimateAdaptive(100, 0.5,0.01, 10, 0.98, 0.5, 0.8, utillg, 10))
