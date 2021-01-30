import numpy as np

# def rbf_sample(gamma_x, gamma_y, N):
#         '''
#         Return random frequcies as in Random Fourier Features. 
        
#         The kernel is defined as 
#                 K = exp(-x^2*gamma_x) * exp(-y^2*gamma_y).
#         The Inverse Fourier transform of K: 
#                 P(w) = C*exp(-0.5*X^T*\Sigma^-1*X),
#         where 
#                 \Sigma = diag([2*gamma_x, 2*gamma_y]).

#         Therefore, we sample from the probability distribution P(w)
#         '''
#         cov = np.diag([2*gamma_x, 2*gamma_y])
#         return np.random.multivariate_normal(np.array([0,0]), cov, N)

def exp_sample(a, N):
        '''
        Return random frequcies as in Random Fourier Features. 
        
        The kernel is defined as 
                K = exp(-abs(x)*a) * exp(-abs(y)*a).
        The Inverse Fourier transform of each separate k: 
                P(w) = 2*gamma/(gamma^2+w^2).

        We sample from P(w) using rejection sampling method.
        '''
        samples = np.zeros((N*2, 2))
        i = 0

        while i < N:
                x,y = np.random.uniform(-400, 400, (2, N))
                p = np.random.uniform(0, 8./(a*a), N)
                u = 4*a*a/((a**2+x**2) * (a**2+y**2))

                mask = p < u
                if mask.sum() > 0:
                        samples[i:i+mask.sum()] = np.hstack([
                                x[mask].reshape((-1,1)), 
                                y[mask].reshape((-1,1))])
                        i += mask.sum()
        return samples[:N]

def exp2_sample(a, N):
        '''
        Return random frequcies as in Random Fourier Features. 
        
        The kernel is defined as 
                K = exp(-a*sqrt(x^2+y^2)).
        This is a special case of Matern class kernel function when nu=1/2.
        '''
        return matern_sample(a, 0.5, N)

def matern_sample(a, nu, N):
        '''
        Return random frequcies as in Random Fourier Features. 
        
        The kernel is Matern class kernel. The Inverse Fourier transform is:
                P(w) = a^{2*nu}/(2*nu*a^2 + ||w||^2)**(n/2+nu).

        We sample from P(w) using rejection sampling method.
        '''
        samples = np.zeros((N*2, 2))
        i = 0

        while i < N:
                x,y = np.random.uniform(-1000, 1000, (2, N))
                p = np.random.uniform(0, 1./((2*nu)**(nu+1) * a**2), N)
                u = a**(2*nu)/(2*nu*a**2 + x**2+y**2)**(nu+1)

                mask = p < u
                if mask.sum() > 0:
                        samples[i:i+mask.sum()] = np.hstack([
                                x[mask].reshape((-1,1)), 
                                y[mask].reshape((-1,1))])
                        i += mask.sum()
        return samples[:N]