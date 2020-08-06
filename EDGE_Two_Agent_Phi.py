
# EDGE Estimator for Shannon Mutual Information
#
# Created by Morteza Noshad (noshad@umich.edu)
# Current version: 4.3.1 
# Requirements: numpy, cvxpy(v1.0.6),scipy, sklearn
#                 
# 10/1/2018
#
# Based on the paper: Scalable Mutual Information Estimation using Dependence Graphs
#
################
# The estimator is in the following form:
#
# I = EDGE(X,Y,U=10, gamma=[1, 1], epsilon=[0,0], epsilon_vector = 'fixed', eps_range_factor=0.1, normalize_epsilon = False ,
#                ensemble_estimation = 'median', L_ensemble=5 ,hashing='p-stable', stochastic = False)
#
# Arguments: 
#
# X is N * d_x and Y is N * d_Y data sets
# U (optional) is an upper bound on the MI. It doesn't need to be accurate, but more accurate upper bound we set, faster convergence rates we get
# gamma=[gamma_X,gamma_Y] (optional) is the vector of soothness for X and Y. 
#        For example, if the data is discrete we set gamma close to 0, 
#        and if the data is continuous we set gamma close to 1 (or maybe higher if it is very smooth) 
# epsilon=[eps_X, eps_Y] (optional) is the vector of bandwidths for X and Y. If no epsilon is set, 
#        automatic bandwidths according to KNN distances will be set.
# epsilon_vector (optional): possible arguments are 'fixed' or 'range'. If 'fixed' is given, all of 
#        the bandwidths for the ensemble estimation will be the same, while, if 'range' is chosen, 
#        the badwidths will be arithmetically increasing in a range.     
# eps_range_factor (optional): If epsilon_vector == 'range', then the range of epsilon is 
#        [epsilon, epsilon*(1+epsilon_vector)].
# normalize_epsilon: If it is True, then the badwidth will be normalized according to the MI estimate 
# ensemble_estimation: several options are available:
#        'average': the ensemble estimator is the average of the base estimators
#        'optimal_weights': the ensemble estimator is the wighted sum of the base estimators
#                            where the weights are computed using an optimization problem
#                            * You need to import cvxpy as cvx (install cvxpy if you do not have it)
#        'median': the ensemble estimator is the median of the base estimators
# L_ensemble: number of different base estimators used in ensemble estimation. For more accurate estimates
#                you can increase L_ensemble, but runtime increases linearly as well.
# hashing (optional): possible arguments are 'p-stable' (default) which is a common type of LSH
#        or 'floor' which uses the simple floor function as hashing. For small dimensions, 'floor', a
#        for higher dimensions, 'p-stable' are preferred.
# stochastic: it is stochastic, the hashing is generated using a random seed.
# 
# Output: I is the estimation of mutual information between X snd Y 
###########################

import numpy as np
import math
import cvxpy as cvx # Need to install CVXPY package, 
                    #  it is also possible to run this code without cvxpy, by 
                    #   using 'average' or 'median' ensemble_estimation
import time
from scipy.special import *
from sklearn.neighbors import NearestNeighbors
import sklearn

#from random import randint, seed
#np.random.seed(seed=0)

#####################
#####################

# Find KNN distances for a number of samples for normalizing bandwidth
def find_knn(A,d):
    np.random.seed(3334)
    #np.random.seed()
    #np.random.seed(seed=int(time.time()))
    r = 500
    # random samples from A
    A = A.reshape((-1,1))
    N = A.shape[0]
    
    k=math.floor(0.43*N**(2/3 + 0.17*(d/(d+1)) )*math.exp(-1.0/np.max([10000, d**4])))
    #print('k,d', k, d)
    T= np.random.choice(A.reshape(-1,), size=r).reshape(-1,1)
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='auto').fit(A)
    distances, indices = nbrs.kneighbors(T)
    d = np.mean(distances[:,-1])
    return d

# Returns epsilon and random shifts b
def gen_eps(XW,YV):
    d_X , d_Y  = XW.shape[1], YV.shape[1]
    # Find KNN distances for a number of samples for normalizing bandwidth
    eps_X = np.array([find_knn(XW[:,[i]],d_X) for i in range(d_X)]) + 0.0001
    eps_Y = np.array([find_knn(YV[:,[i]],d_Y) for i in range(d_Y)]) + 0.0001

    return (eps_X,eps_Y)

# Define H1 (LSH) for a vector X (X is just one sample)
def H1(XW,b,eps):
    
    # dimension of X
    d_X = XW.shape[0]
    #d_W = W.shape[1]
    XW=XW.reshape(1,d_X)

    # If not scalar
    if d_X > 1:
        X_te = 1.0*(np.squeeze(XW)+b)/eps    
    elif eps>0:
        X_te = 1.0*(XW+b)/eps
    else:
        X_te=XW

    # Discretize X
    X_t = np.floor(X_te)
    if d_X>1: 
        R = tuple(X_t.tolist())
    else: R=np.asscalar(np.squeeze(X_t))
    return R

# Compuate Hashing: Compute the number of collisions in each bucket
def Hash(XW,YV,eps_X,eps_Y,b_X,b_Y):

    # Num of Samples and dimensions
    N = XW.shape[0]

    # Hash vectors as dictionaries
    CX, CY, CXY = {}, {}, {} 
    
    # Computing Collisions
    
    for i in range(N):
        # Compute H_1 hashing of X_i and Y_i: Convert to tuple (vectors cannot be taken as keys in dict)

        X_l, Y_l = H1(XW[i],b_X,eps_X), H1(YV[i],b_Y,eps_Y)

        # X collisions: compute H_2 
        if X_l in CX:
            CX[X_l].append(i)
        else: 
            CX[X_l] = [i]
            
        # Y collisions: compute H_2
        if Y_l in CY:
            CY[Y_l].append(i)
        else: 
            CY[Y_l] = [i]

        # XY collisions
        if (X_l,Y_l) in CXY:
            CXY[(X_l,Y_l)].append(i)
        else: 
            CXY[(X_l,Y_l)] = [i]

    return (CX, CY, CXY)



### Phi parameter calculations
def calc_distance(XW, YV, N):
    T = XW.shape[0]
    t = T-N
    sigma = 0
    dist_array = np.zeros([5])
    n = 0
    for i in range(t,T):
        dist_array[n] = ((XW[i][0]-YV[i][0])**2 + (XW[i][1]-YV[i][1])**2)**(1/2)
        sigma += dist_array[n]
        n += 1
    distance = sigma/N
    
    sigma = 0
    for i in range(1, len(dist_array)):
            ### The change in distance between each frame in the window N
            conv = dist_array[i] - dist_array[i-1]
            sigma += conv
    convergence = sigma/(N-1)

    return distance, convergence

    
def calc_speed(XW, YV, N):
    T = XW.shape[0]
    t = T-N
    sigma = 0
    vel_x = np.zeros(5)
    vel_y = np.zeros(5)
    n = 0
    ### This is never called when i=0, due to our approach of using T_prime as a buffer window
    for i in range(t,T):
        ### Distance between XW[i] and the previous point of XW[i-1]. Since time is 0.4 seconds per frame, 
        ###     this represents a measure of velocity in units per second, could alternatively omit the 2.5
        
        vel_x[n] = ((XW[i][0]-XW[i-1][0])**2 + (XW[i][1]-XW[i-1][1])**2)**(1/2) * 2.5
        vel_y[n] = ((YV[i][0]-YV[i-1][0])**2 + (YV[i][1]-YV[i-1][1])**2)**(1/2) * 2.5
        sigma +=  vel_x[n] + vel_y[n]
        n += 1
    velocity = sigma/N

    sigma = 0
    for i in range(1, len(vel_x)):
            ### using acceleration = d.velocity/d.time, time is left out since its at 1 second intervals now
            acc_x = abs(vel_x[i] - vel_x[i-1])
            acc_y = abs(vel_y[i] - vel_y[i-1])
            sigma += acc_x + acc_y
    acceleration = sigma/(N-1) 
    
    return velocity, acceleration









def mi_t_edge(data, phi_type = '', U=10, gamma=[1, 1, 1], epsilon=[0,0,0], epsilon_vector = 'range', eps_range_factor=0.1, normalize_epsilon = True, ensemble_estimation = 'average', L_ensemble=10, hashing='p-stable', cmi='cmi3'):


    ### Treats N as time and 2nd dimension as 2-tuple coordinates
    if(data.shape[0] == 2 and data[0].shape[0] == data[1].shape[0]):
        T = data[0].shape[0]
        I_T = 0;
        
        ### This should go from 1 to T, but I'm leaving it as is for now. Would likely need to
        ###    include an exception for t==1 since z would have to be empty, which shouldnt work with
        ###    the rest of the code. I think it can be assumed that I for two single numbers is negligible though.
        if(T > 200):
            ### Uses T' of 20 frames as a buffer for analysis
            for t in range(201, T):    
                x = data[0][:t , :].reshape(t, 2) 
                y = data[1][:t , :].reshape(t, 2) 
                
                #Already checked that this is properly passing the phi_type
                I_value = max(0, EDGE(x,y, phi_type = phi_type, hashing = hashing))
                print("     Time ", t)
                print("     Value: ",I_value)
                I_T +=I_value
            
            end = time.time()
            print("Running ", n_samples, " samples.")
            print("Value is: ", I_T)
        else:
            print("Insufficient time points for T prime")            
    else:
        print("Incompatible data structure, supply a 2xNx2 tensor")
        






def EDGE(X,Y, phi_type = "", U=10, gamma=[1, 1], epsilon=[0,0], epsilon_vector = 'range', eps_range_factor=0.1, 
         normalize_epsilon = True, ensemble_estimation = 'average', L_ensemble=10 ,hashing='p-stable', stochastic = False):
    
    gamma = np.array(gamma)
    gamma = gamma * 0.4
    epsilon = np.array(epsilon)

    if X.ndim==1:
        X=X.reshape((-1,1))
    if Y.ndim==1:
        Y=Y.reshape((-1,1))
    # Num of Samples and dim
    N, d = X.shape[0], X.shape[1]
    dy = Y.shape[1]
    
    # Find dimensions
    dim_X, dim_Y  = X.shape[1], Y.shape[1]


## Hash type

    if hashing == 'floor':
        d_X_shrink, d_Y_shrink = dim_X, dim_Y 
        XW, YV = X, Y
    
## Initial epsilon and apply smoothness gamma

    # If no manual epsilon is set for computing MI:
    if epsilon[0] ==0:
        # Generate auto epsilon and b
        (eps_X_temp,eps_Y_temp) = gen_eps(XW,YV)
        # Normalizing factors for the bandwidths
        cx, cy = 18*d_X_shrink / np.max([(1+1.*math.log(dim_X)),1]), 18*d_Y_shrink/ np.max([(1+1.*math.log(dim_Y)),1])
        eps_X0, eps_Y0 = eps_X_temp * cx*gamma[0], eps_Y_temp * cy*gamma[1] 
        ##### At this point eps_X0 and Y0 have been derived from knn run on each dimension, and cx and cy only take into account
        ##### The number of dimensions, so it was identical for x and y, but since the knn differed slightly, eps is different still
    else:
        eps_X_temp = np.ones(d_X_shrink,)*epsilon[0]
        b_X = np.linspace(0,1,L_ensemble,endpoint=False)[j]*eps_X
        b_Y = np.linspace(0,1,L_ensemble,endpoint=False)[j]*eps_Y
        eps_Y_temp = np.ones(d_Y_shrink,)*epsilon[1]    
        cx, cy = 15*d_X_shrink / np.max([(1+1.0*math.log(dim_X)),1]), 15*d_Y_shrink/ np.max([(1+1.0*math.log(dim_Y)),1])
        eps_X0, eps_Y0 = eps_X_temp * cx*gamma[0], eps_Y_temp * cy*gamma[1] 

    ## epsilon_vector
    T = np.linspace(1,1+eps_range_factor,L_ensemble)        



## Compute MI Vector
    # MI Vector
    I_vec = np.zeros(L_ensemble)
    for j in range(L_ensemble):

        # Apply epsilon_vector 
        eps_X, eps_Y = eps_X0 * T[j], eps_Y0 * T[j]

        b_X = np.linspace(0,1,L_ensemble,endpoint=False)[j]*eps_X
        b_Y = np.linspace(0,1,L_ensemble,endpoint=False)[j]*eps_Y

        I_vec[j] = Compute_MI(XW,YV,phi_type,U,eps_X,eps_Y,b_X,b_Y)

## Ensemble method
    if ensemble_estimation == 'average':
        I = np.mean(I_vec)
    elif ensemble_estimation == 'median':
        I = np.median(I_vec)

## Normalize epsilon according to MI estimation (cross validation)
    if normalize_epsilon == True:
        gamma=gamma * math.pow(2,-math.sqrt(I*2.0)+(0.5/I))
        normalize_epsilon = False
        I = EDGE(X,Y, phi_type, U, gamma, epsilon, epsilon_vector, eps_range_factor, normalize_epsilon, ensemble_estimation, L_ensemble,hashing, stochastic)

    return I








# Compute mutual information and gradient given epsilons and radom shifts
def Compute_MI(XW,YV, phi_type, U,eps_X,eps_Y,b_X,b_Y):
    N = XW.shape[0]

    (CX, CY, CXY) = Hash(XW,YV,eps_X,eps_Y,b_X,b_Y)
    
    # Computing Mutual Information Function
    I = 0
    J = 0
    N_c = 0
    for e in CXY.keys():
        ### e is a 1x2x2 tensor, a pair of hashed coordinates
        Ni = len(CX[e[0]])
        Nj = len(CY[e[1]])
        Nij = len(CXY[e])


        #Determines the phi being used when determining I
        ### If phi isnt being based on the keys in e, then this should only be done once rather than in a for loop
        if phi_type == 'act' or phi_type == 'act_wt':
            phi = np.linalg.norm(e[0])*np.linalg.norm(e[1])*np.linalg.norm(e[2])

        elif phi_type == 'e_act' or phi_type == 'e_act_wt':
            phi = np.exp(-(np.linalg.norm(e[0])*np.linalg.norm(e[1])*np.linalg.norm(e[2]))**2 / 2.)

        elif phi_type == 'act_sq' or phi_type == 'act_wt_sq':
            phi = (np.linalg.norm(e[0])**2)*(np.linalg.norm(e[1])**2)*(np.linalg.norm(e[2])**2)

        ### Not finished, doesnt normalize in a fully sensible way
        elif phi_type == 'dvac':
            W = 5
            dis, conv = calc_distance(XW, YV, W)
            vel, acc = calc_speed(XW, YV, W)
            phi = max(0, dis*vel*acc*conv)
        else:
            phi = 1.
            
        ### Taken from XYZ ADI Version, g_func is ((t-1)**2/(2*(t+1))) in that code for some reason
        ### num   = phi * (Nik*Njk) * g_func(Nijk*Nk/(Nik*Njk)) 
        ### denom = Ni
        
        ### Doesnt currently actually use phi since the function isnt complete
            
        ### g() is this bounded log function
        ### But why is wi wj replaced with Nij here?
        ### Multiplying by Nij and then dividing by the sum of Nij is done in order to get a weighted average, where sum(Nij) = N
        I += Nij* max(min(math.log(1.0*Nij*N/(Ni*Nj),2), U),0.001)
        N_c+=Nij

    I = 1.0* I / N_c
    
    return I





####################################
####################################
if __name__ == "__main__":


    ### Makes 2 Nx2 datasets as placeholders for 500 frame trajectories    
    n_samples = 500
    np.random.seed(seed=int(time.time()))

    X = np.random.multivariate_normal(mean=np.zeros((2,)), cov=np.identity(2), size=(n_samples))
    Y = np.random.multivariate_normal(mean=np.zeros((2,)), cov=np.identity(2), size=(n_samples))
    data = np.array([X,Y])

    
    #I = EDGE(X,Y, phi_type = "dvac", hashing="floor")
    mi_t_edge(data, phi_type = 'dvac', hashing = 'floor')

    #print ('Estimated MI',I)
    print('################################')
    
