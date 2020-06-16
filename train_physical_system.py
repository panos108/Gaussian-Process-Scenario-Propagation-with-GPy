import GPy
from pyDOE import *
from casadi import *
import scipy
import time
import sobol_seq
import pickle
import matplotlib.pyplot as plt
class GP_c():
    def __init__(self, casadi, info_integration, normalization = 'Normal',save=False):
        if casadi:
            self.specifications, self.DAE_system, self.integrator_model = info_integration
            self.nk, self.tf, self.x0, self.Lsolver, self.c_code = self.specifications()
            self.xd, self.xa, self.u, self.uncertainty, self.ODEeq, self.Aeq, self.u_min, self.u_max, self.states, \
            self.algebraics, self.inputs, self.nd, self.na, self.nu, self.nmp, self.modparval = self.DAE_system()
        else:
            raise NotImplementedError
        self.normalization = normalization
        self.save = save
    def Generate_data(self, N_exp):

        F = self.integrator_model()
        nk, tf, x0, Lsolver, c_code = self.nk, self.tf, self.x0, self.Lsolver, self.c_code
        xd, xa, u, uncertainty, ODEeq, Aeq, u_min, u_max, states, algebraics, inputs, nx,\
        na, nu, nmp, modparval  = self.xd, self.xa, self.u, self.uncertainty, self.ODEeq, \
                                  self.Aeq, self.u_min, self.u_max, self.states, self.algebraics,\
                                  self.inputs, self.nd, self.na, self.nu, self.nmp, self.modparval

        x0_max = np.zeros(nx)
        x0_min = np.zeros(nx)
        Sigma_v = [400.,1e5,1e-2]*diag(np.ones(nx))*1e-6
        for i in range(nx):
            if x0[i] >1e-3:
                x0_max[i] = x0[i]*1.1
                x0_min[i] = x0[i]*0.9


        N     = N_exp
        set_u = 2*sobol_seq.i4_sobol_generate(nu,N*nk)-1#2 * lhs(nu, samples=N*nk) - 1
        set_x = 2*sobol_seq.i4_sobol_generate(nx,N*nk)-1
        # ------- Transform the normalized variables to the real ones -----

        range_u = np.array([u_min, u_max]).T
        range_x = np.array([x0_min, x0_max]).T

        x0_t    = (range_x.max(axis=1) - range_x.min(axis=1)) / 2 * set_x \
                  + (range_x.max(axis=1) + range_x.min(axis=1)) / 2

        u_t = (range_u.max(axis=1) - range_u.min(axis=1)) / 2 * set_u \
              + (range_u.max(axis=1) + range_u.min(axis=1)) / 2

        u_t  = u_t.T
        x0_t = x0_t.T
        x    = np.zeros([nx, N*nk])
        xin  = np.zeros([nx, N*nk])

        for i in range(N):
            x0 = x0_t[:,i]
            integrated_state = x0
            for k in range(nk):



                xin[:, i*nk + k] = integrated_state
                xd = F(x0=vertcat(np.array(integrated_state)), p=vertcat(u_t[:,i*nk + k], np.array(modparval)*0.))
                integrated_state = np.array(xd['xf'].T)[0]+np.random.multivariate_normal([0.]*nx,np.array(Sigma_v))
                for ii in range(nx):
                    if integrated_state[ii]<0:
                        integrated_state[ii] = 0
                x[:, i*nk + k] = integrated_state+np.random.multivariate_normal([0.]*nx,np.array(Sigma_v)*0.1)
                for ii in range(nx):
                    if x[ii, i*nk + k] < 0:
                        x[ii, i*nk + k] = 0
        """
        Generate data-set
        """







        uu = np.vstack((xin, u_t))
        # GP = GP_model(uu.T[:300,:], s.T[:300,:], 'RBF', 20, [])
        if self.normalization == 'Uniform':
            # kb = GPy.kern.Brownian(input_dim=uu.shape[0])

            mean_u =  (uu.max(axis=1).reshape((-1, 1)) + uu.min(axis=1).reshape((-1, 1)))/2
            std_u  =  (uu.max(axis=1).reshape((-1, 1)) - uu.min(axis=1).reshape((-1, 1)))/2
            mean_x =  (x.max(axis=1).reshape((-1, 1)) + x.min(axis=1).reshape((-1, 1)))/2
            std_x  =  (x.max(axis=1).reshape((-1, 1)) - x.min(axis=1).reshape((-1, 1)))/2


        else:
            mean_u = uu.mean(axis=1).reshape((-1, 1))
            std_u  = uu.std(axis=1).reshape((-1, 1))
            mean_x = x.mean(axis=1).reshape((-1, 1))
            std_x  = x.std(axis=1).reshape((-1, 1))
        self.mean_x, self.std_x, self.mean_u, self.std_u = mean_x, std_x, mean_u, std_u
        self.u_un= uu
        self.u_n = (uu - mean_u) / std_u
        self.x_n = (x - mean_x) / std_x

        return self.mean_x, self.std_x, self.mean_u, self.std_u, self.u_n, self.x_n, uu, x

    def GP_regression(self):
        self.GP_p    = []
        self.GP_full = []
        self.invK    =[]
        for i in range(self.nd):
            kernel = GPy.kern.Linear(input_dim=self.u_n.shape[0],
                                    ARD=True) + GPy.kern.RBF(input_dim=self.u_n.shape[0],
                                    ARD=True)
            # GP_f = GPy.models.SparseGPRegression(self.u_n.T, self.x_n.T[:,i].reshape(-1,1), kernel)
            # GP_f.randomize()
            # GP_f.Z.unconstrain()
            # GP_f.constrain_bounded(1e-8, 1e2)
            # GP_f.optimize_restarts(num_restarts=10)

            GP_f = GPy.models.GPRegression(self.u_n.T, self.x_n.T[:,i].reshape(-1,1), kernel)
            GP_f.constrain_bounded(1e-8, 1e2)
            GP_f.optimize_restarts(num_restarts=10)
            self.GP_p.append(GP_f.predict)  # (lamda[0].reshape((-1,1)))
            self.GP_full.append(GP_f)
            sigma2 = np.array(GP_f.likelihood.parameters)[0][0]# + 1e-8#model.predict_noiseless(u_s.reshape((1,-1)))

            K = GP_f.kern.K((self.u_n.T)) \
                + sigma2 * scipy.linalg.block_diag(np.eye(self.u_n.shape[1])) \
                + 1e-8 * np.eye(self.u_n.shape[1])
            K = (K+K.T)*0.5
            self.invK.append(np.linalg.solve(K,np.eye(self.u_n.shape[1])))
            if self.save:
                pickle.dump([self.GP_full, self.GP_p], open("GP_ISbRwSc.p", "wb"))
        return self.GP_full, self.GP_p

    def predict_GP(self, u):

        u_n     = (u.reshape((-1,1)) - self.mean_u) / self.std_u
        mu_n    = np.zeros(self.nd)
        sigma_n = np.zeros(self.nd)
        for i in range(self.nd):
            mu_n[i], sigma_n[i] = self.GP_p[i](u_n.reshape((1,-1)))
        mu    = mu_n*self.std_x.reshape((-1,)) + self.mean_x.reshape((-1,))
        sigma = sigma_n * self.std_x.reshape((-1,)) **2# self.std_x.reshape(1, self.nd))

        return mu, sigma


    def predict_GP_norm(self, u):
        u_n     = (u.reshape((-1,1)) - self.mean_u) / self.std_u
        mu_n    = np.zeros(self.nd)
        sigma_n = np.zeros(self.nd)
        for i in range(self.nd):
            mu_n[i], sigma_n[i] = self.GP_p[i](u_n.reshape((1,-1)))
        return mu_n, sigma_n

    def Online_MatrixInv(self, Conv_inv, k, i):#xnorm, Xsample, i):
        # array manipulation
        #k   = self.GP_full[i].kern.K(xnorm, Xsample)

        A22 = np.array([self.GP_full[i].kern.parameters[0][0]])#np.array(self.GP_full[i].likelihood.parameters)[0][0]+1e-8])
        A22 = A22.reshape((1,1))
        A12 = k.reshape((k.shape[0], 1))
        A21 = k.reshape((1,k.shape[0]))
        I   = Conv_inv
        II  = np.matmul(A21,I)
        III = np.matmul(I,A12)
        IV  = np.matmul(A21,III)
        V   = IV - A22
        VI  = 1./V
        C12 = III * VI
        C21 = VI * II
        VII = np.matmul(III,C21)
        C11 = I - VII
        C22 = -VI
        C   = np.block([[C11,C12],[C21,C22]])

        return C


    def predict_GP_based_scenario(self, u, U_, X_):

        u_s = (u.reshape((-1,1)) - self.mean_u) / self.std_u
        U_n = (U_ - self.mean_u) / self.std_u
        X_n = (X_ - self.mean_x) / self.std_x
        U   = np.hstack((self.u_n, U_n))
        X   = np.hstack((self.x_n, X_n))



        mu_n    = np.zeros(self.nd)
        sigma_n = np.zeros(self.nd)
        start = time.time()

        for i in range(self.nd):
            #model = GPy.models.GPRegression(U.T, X.T[:,i].reshape((-1,1)), self.GP_full[i].kern)
            #mu_n[i], sigma_n[i] =  self.GP_p[i](u_s.reshape((1,-1)))
            sigma2 = np.array(self.GP_full[i].likelihood.parameters)[0][0]# + 1e-8#model.predict_noiseless(u_s.reshape((1,-1)))
            K   = self.GP_full[i].kern.K((U.T))\
                  +(sigma2+1e-8)*scipy.linalg.block_diag(np.eye(self.u_n.shape[1]), *np.zeros(U_n.shape[1]))#+1e-8*np.eye(U.shape[1])


            K   = (K + K.T) * 0.5
            k_s = self.GP_full[i].kern.K(U.T, u_s.reshape((1,-1)))
            L = np.linalg.cholesky(K)


            invLY   = np.linalg.solve(L,  X.T[:,i].reshape((-1,1)))
            alpha   = np.linalg.solve(L.T, invLY)
            v       = np.linalg.solve(L, k_s)
            mu_n[i] = k_s.T@alpha
            sigma_n[i]= self.GP_full[i].kern.K(u_s.reshape((1,-1)),u_s.reshape((1,-1))) - v.T@v


        mu    = mu_n*self.std_x.reshape((-1,)) + self.mean_x.reshape((-1,))
        sigma = sigma_n * self.std_x.reshape((-1,)) **2# self.std_x.reshape(1, self.nd))
        print('old',time.time() - start)
        return mu, sigma


    def predict_GP_based_scenario_test(self, u, U_, X_, invK):

        u_s = (u.reshape((-1,1)) - self.mean_u) / self.std_u
        U_n = (U_ - self.mean_u) / self.std_u
        X_n = (X_ - self.mean_x) / self.std_x
        U   = np.hstack((self.u_n, U_n))
        X   = np.hstack((self.x_n, X_n))
        U_up = U_n[:,-1].reshape((-1,1))

        mu_n    = np.zeros(self.nd)
        sigma_n = np.zeros(self.nd)

        for i in range(self.nd):

            k_s     = self.GP_full[i].kern.K(U.T, u_s.reshape((1,-1)))
            k_s_old = self.GP_full[i].kern.K(U[:,:-1].T, U_up.reshape((1,-1)))
            invK[i] = self.Online_MatrixInv(invK[i], k_s_old, i)

            mu_n[i] = k_s.T@invK[i]@ X.T[:,i].reshape((-1,1))#alpha
            sigma_n[i]= self.GP_full[i].kern.K(u_s.reshape((1,-1)),u_s.reshape((1,-1))) - k_s.T@invK[i]@k_s#v.T@v





        mu    = mu_n*self.std_x.reshape((-1,)) + self.mean_x.reshape((-1,))
        sigma = sigma_n * self.std_x.reshape((-1,)) **2# self.std_x.reshape(1, self.nd))
        sigma = np.maximum(sigma, [1e-10] * self.nd)

        return mu, sigma, invK


class GPR(Callback):
    def __init__(self, name, opts={}):
        Callback.__init__(self)
        self.construct(name, opts)

    def eval(self, arg):
        [mean, _] = model.predict_y(np.array(arg[0]))
        return [mean]

from Dynamic_system import integrator_model, specifications, DAE_system

def training_physical():
    info_integration =  [specifications, DAE_system, integrator_model ]
    GP = GP_c(casadi, info_integration, normalization = 'Normal', save=True)
    nx = GP.nd
    nu = GP.nu
    mean_x, std_x, mean_u, std_u, u_n, x_n, uu, x = GP.Generate_data(5)
    GP_full, GP_p = GP.GP_regression()
    GP.predict_GP_norm(mean_u)
    GP.predict_GP(GP.u_un.T[0,:])
    N =12
    MC= 1000
    his_u      = np.zeros([nu+nx,N+1,MC])
    his_x      = np.zeros([nx,N+1,MC])
    his_sigma  = np.zeros([nx,N+1,MC])
    his_mean   = np.zeros([nx,N+1,MC])
    his_u_real = np.zeros([nu+nx,N+1,MC])
    his_x_real = np.zeros([nx,N+1,MC])
    start = time.time()
    set_u = 2 * sobol_seq.i4_sobol_generate(nu, MC * 12) - 1  # 2 * lhs(nu, samples=N*nk) - 1
    range_u = np.array([[120, 0], [400, 40]]).T

    u_t = (range_u.max(axis=1) - range_u.min(axis=1)) / 2 * set_u \
          + (range_u.max(axis=1) + range_u.min(axis=1)) / 2
    u_t = u_t.T

    for kk in range(MC):
        u = GP.u_un.T[0,:].copy()
        invK = GP.invK.copy()
    #    start1 = time.time()
        his_x[:, 0, kk] = u[:nx]
        his_u[:, 0, kk] = u
        his_mean[:, 0, kk] =  u[:nx]
        his_sigma[:, 0, kk] = u[:nx]*0

        for i in range(12):
            if i>=0:
                mu,sigma = GP.predict_GP(u)



            else:
                # mu,sigma = GP.predict_GP(u)#predict_GP_based_scenario(u, u_old, u[:-1])    start = time.time()
                mu, sigma, invK = GP.predict_GP_based_scenario_test(u, his_u[:, :i,kk], his_x[:, :i,kk],invK)#u_old, u[:-1])
                #mu, sigma = GP.predict_GP_based_scenario(u, his_u[:, :i,kk], his_x[:, :i,kk])#u_old, u[:-1])

            u[-nu:] = u_t[:,kk*12 + i]
            u_old = u.copy()
            u[:nx] = np.random.multivariate_normal(mu,np.diagflat(sigma))
            for j in range(len(mu)):
                if u[j] < 0:
                    u[j] = 0.
            his_x[:, i+1, kk] = u[:nx]
            his_u[:, i+1, kk] = u_old
            his_mean[:, i, kk] = mu
            his_sigma[:, i, kk]= sigma

    #    elapsed_time_fl = (time.time() - start1)
    #    print('total:', elapsed_time_fl)
    elapsed_time_f2 = (time.time() - start)
    print('total:', elapsed_time_f2)
    start = time.time()
    modparval = np.array([0.0923*0.62, 178.85, 447.12, 393.10, 0.001, 504.49,
    2.544*0.62*1e-4, 23.51, 800.0, 0.281, 16.89])*0.

    for kk in range(MC):

        F = GP.integrator_model()
        integrated_state = GP.u_un.T[0, :nx].copy()


        his_x_real[:, 0, kk] = his_x[:, 0, kk]
        his_u_real[:, 0, kk] = his_u[:, 0, kk]
        for i in range(N):

            Sigma_v = [400., 1e5, 1e-2] * diag(np.ones(nx)) * 1e-6
            xd = F(x0=vertcat(np.array(integrated_state)), p=vertcat(his_u[-2:, i, kk], modparval))
            integrated_state = np.array(xd['xf'].T)[0] #+np.random.multivariate_normal([0.]*nx,np.array(Sigma_v))
            x = integrated_state#+np.random.multivariate_normal([0.]*nx,np.array(Sigma_v)*0.1)
            for ii in range(nx):
                if integrated_state[ii] < 0:
                    integrated_state[ii] = 0
            for ii in range(nx):
                if x[ii] < 0:
                    x[ii] = 0

            his_x_real[:, i+1, kk] = x
            his_u_real[:, i+1, kk] = his_u[:, i, kk]
    elapsed_time_f3 = (time.time() - start)
    print('total:', elapsed_time_f3)

    return GP
print(0)
