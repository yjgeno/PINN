import numpy as np
import tensorflow as tf
from pyDOE import lhs  # for latin hypercube sampling


class ODE_data_generator():
    def __init__(self, nu = 0.01/np.pi):
        self.nu = nu # viscosity parameter
        # define grid for quadrature solution
        self.utn = 128
        self.uxn = 256
        self.xlo = -1.0
        self.xhi = +1.0  
        self.tlo = 0.0
        self.thi = 5.0/np.pi
        self._X_flat, self._u_flat = self.get_num_solution()

    def get_num_solution(self):
        from numpy.polynomial.hermite import hermgauss
        ux = np.linspace(self.xlo,self.xhi,self.uxn)
        ut = np.linspace(self.tlo,self.thi,self.utn)

        qn = 64 # order of quadrature rule
        qx,qw = hermgauss(qn)

        # compute solution u(x,t) by quadrature of analytical formula:
        u_quad = np.zeros([self.uxn,self.utn])
        for utj in range(self.utn):
            if (ut[utj]==0.0):
                for uxj in range(self.uxn):
                    u_quad[uxj,utj] = -np.sin(np.pi*ux[uxj])
            else:
                for uxj in range(self.uxn):
                    top = 0.0
                    bot = 0.0
                    for qj in range(qn):
                        c = 2.0*np.sqrt(self.nu*ut[utj])
                        top = top-qw[qj]*c*np.sin(np.pi*(ux[uxj]-c*qx[qj]))*np.exp(-np.cos(np.pi*(ux[uxj]-c*qx[qj]))/(2.0*np.pi*self.nu))
                        bot = bot+qw[qj]*c*np.exp(-np.cos(np.pi*(ux[uxj]-c*qx[qj]))/(2.0*np.pi*self.nu))
                        u_quad[uxj,utj] = top/bot

        # flatten grid and solution
        X,T = np.meshgrid(ux,ut)
        _X_flat = tf.convert_to_tensor(np.hstack((X.flatten()[:,None],T.flatten()[:,None])),dtype=tf.float32)
        _u_flat = tf.convert_to_tensor(u_quad.T.flatten(),dtype=tf.float32)
        return _X_flat, _u_flat
    
    @property
    def X_flat(self):
        return self._X_flat

    @property
    def u_flat(self):
        return self._u_flat

    # data points
    def get_data_points(self, Ns=10000):
        idxs = tf.range(tf.shape(self.X_flat)[0])
        ridxs = tf.random.shuffle(idxs)[:Ns]
        xs = tf.expand_dims(tf.gather(self.X_flat[:,0],ridxs),-1)
        ts = tf.expand_dims(tf.gather(self.X_flat[:,1],ridxs),-1)
        us = tf.expand_dims(tf.gather(self.u_flat,ridxs),-1)
        return xs, ts, us

    # collocation points
    def get_collocation_points(self, Ncl=10000):
        X = lhs(2,Ncl)
        xcl = tf.expand_dims(tf.convert_to_tensor(self.xlo+(self.xhi-self.xlo)*X[:,0],dtype=tf.float32),-1)
        tcl = tf.expand_dims(tf.convert_to_tensor(self.tlo+(self.thi-self.tlo)*X[:,1],dtype=tf.float32),-1)
        return xcl, tcl

    # Dirichlet boundary condition points
    def get_BC_points(self, Nlb=500, Nub=500):
        X = lhs(1,Nlb)
        tlb = tf.expand_dims(tf.convert_to_tensor(self.tlo+(self.thi-self.tlo)*X[:,0],dtype=tf.float32),-1)
        xlb = self.xlo*tf.ones(tf.shape(tlb),dtype=tf.float32)
        ulb = tf.zeros(tf.shape(tlb),dtype=tf.float32)

        X = lhs(1,Nub)
        tub = tf.expand_dims(tf.convert_to_tensor(self.tlo+(self.thi-self.tlo)*X[:,0],dtype=tf.float32),-1)
        xub = self.xlo*tf.ones(tf.shape(tub),dtype=tf.float32)
        uub = tf.zeros(tf.shape(tub),dtype=tf.float32)
        return xlb, tlb, ulb, xub, tub, uub

    # test points for initial condition
    def get_test_points(self, N0=500):
        X = lhs(1,N0)
        x0 = tf.expand_dims(tf.convert_to_tensor(self.xlo+(self.xhi-self.xlo)*X[:,0],dtype=tf.float32),-1)
        t0 = tf.zeros(tf.shape(x0),dtype=tf.float32)
        u0 = -tf.math.sin(np.pi*x0)
        return x0, t0, u0

def get_generator(gen, **kwargs):
    xs, ts, us = gen.get_data_points(**kwargs)
    xcl, tcl = gen.get_collocation_points(**kwargs)
    xlb, tlb, ulb, xub, tub, uub = gen.get_BC_points(**kwargs)
    x0, t0, u0 = gen.get_test_points(**kwargs)

    X_flat, u_flat = gen.X_flat, gen.u_flat 

    return [xcl,tcl,xs,ts,us,xlb,tlb,ulb,xub,tub,uub,x0,t0,u0], [X_flat, u_flat]

if __name__ == '__main__':
    import sys
    gen = ODE_data_generator()
    gen_points = get_generator(gen)
    names = ['xcl','tcl','xs','ts','us','xlb','tlb','ulb','xub','tub','uub','x0','t0','u0']
    for p, name in zip(gen_points[0], names):
        print(f'{name}: {len(p)}\n')  # [10000, 10000, 10000, 10000, 10000, 500, 500, 500, 500, 500, 500]
    sys.exit()
