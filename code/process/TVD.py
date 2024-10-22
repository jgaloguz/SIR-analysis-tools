# This code contains the total variation diminising routine with a maximization-minimization flavor.

import numpy as np
from scipy.sparse import diags
from scipy.linalg import solve_banded

def TVDmm(y, lam, tol, Niter):
   """
   Total Variation Denoising (Majorization-Minimization) algorithm.
    
   INPUT:
      y - noisy data (should be numpy array)
      lam - regularization parameter
      tol - tolerance to stop iterations
      Niter - max number of iterations
   
   OUTPUT:
      x - denoised data
      cost - cost function history

   Adapted from MATLAB code by Ivan Selesnick in "Total Variation Denoising (an MM algorithm)" (2017)
   https://eeweb.engineering.nyu.edu/iselesni/lecture_notes/TVDmm/TVDmm.pdf
   """

   N = np.size(y) # Size of array
   D = diags([np.full(N-1,-1.0), np.full(N-1,1.0)], offsets=[0,1], shape=(N-1,N)) # Sparse (N-1xN) D matrix
   DT = diags([np.full(N-1,-1.0), np.full(N-1,1.0)], offsets=[0,-1], shape=(N,N-1)) # Sparse (NxN-1) D^T matrix
   F = np.empty((3,N-1)) # Sparse (N-1xN-1) triagonal matrix (1/lambda)*diag(|Dx|) + D*D^T
   F[0,:] = -1.0 # Upper diagonal (always -1)
   F[2,:] = -1.0 # Lower diagonal (always -1)

   x = y # Initial guess
   Dx = D.dot(x) # Sparse (N-1) D*x vector
   Dy = D.dot(y) # Sparse (N-1) D*y vector

   cost = np.zeros(Niter)
   k = 0
   while k < Niter:
      F[1,:] = 2.0 + np.abs(Dx)/lam # Main diagonal of F (variable because of Dx)
      x = y - DT.dot(solve_banded((1,1), F, Dy)) # Solve banded linear system
      Dx = D.dot(x) # Sparse (N-1) D*x vector
      cost[k] = 0.5*np.linalg.norm(x-y)**2 + lam*np.linalg.norm(Dx,1) # Cost function (i.e. function being minimized)
      if np.abs(cost[k] - cost[k-1]) / (np.abs(cost[k]) + np.abs(cost[k-1])) < tol:
         break
      k = k+1

   if k == Niter:
      print("\tWARNING: TVDmm did not converge in", Niter, "iterations with a tolerance of", tol)

   return x, cost[0:k]

def TVDalt(y, lam):
   """
   Total Variation Denoising (alternative) algorithm.
    
   INPUT:
      y - noisy data (should be numpy array)
      lam - regularization parameter
   
   OUTPUT:
      x - denoised data

   This algorithm minimizes a cost function containing the term lam/2 * || Dx ||_2^2 instead of lam * || Dx ||_1
   """

   N = np.size(y) # Size of array
   F = np.empty((3,N)) # Sparse (NxN) triagonal matrix I + lam * D*D^T
   F[0,:] = -1.0 # Upper diagonal (always -1)
   F[1,0] = F[1,N-1] = 1.0 + lam # Main diagonal (first & last element)
   F[1,1:N-1] = 1.0 + lam * 2.0 # Main diagonal (other elements)
   F[2,:] = -1.0 # Lower diagonal (always -1)

   return solve_banded((1,1), F, y)