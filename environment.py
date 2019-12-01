
import numpy as np
from scipy.integrate import ode
import unittest

from asm2d import *

# 0	Influent	S_A
# 1	Influent	S_ALK
# 2	Influent	S_F
# 3	Influent	S_I
# 4	Influent	S_N2
# 5	Influent	S_NH4
# 6	Influent	S_NO3
# 7	Influent	S_O2
# 8	Influent	S_PO4
# 9	Influent	X_AUT
# 10	Influent	X_H
# 11	Influent	X_I
# 12	Influent	X_PAO
# 13	Influent	X_PHA
# 14	Influent	X_PP
# 15	Influent	X_S
# 16	Influent	X_TSS
# 17	Anaerobic	S_A
# 18	Anaerobic	S_ALK
# 19	Anaerobic	S_F
# 20	Anaerobic	S_I
# 21	Anaerobic	S_N2
# 22	Anaerobic	S_NH4
# 23	Anaerobic	S_NO3
# 24	Anaerobic	S_O2
# 25	Anaerobic	S_PO4
# 26	Anaerobic	X_AUT
# 27	Anaerobic	X_H
# 28	Anaerobic	X_I
# 29	Anaerobic	X_PAO
# 30	Anaerobic	X_PHA
# 31	Anaerobic	X_PP
# 32	Anaerobic	X_S
# 33	Anaerobic	X_TSS
# 34	Anoxic	S_A
# 35	Anoxic	S_ALK
# 36	Anoxic	S_F
# 37	Anoxic	S_I
# 38	Anoxic	S_N2
# 39	Anoxic	S_NH4
# 40	Anoxic	S_NO3
# 41	Anoxic	S_O2
# 42	Anoxic	S_PO4
# 43	Anoxic	X_AUT
# 44	Anoxic	X_H
# 45	Anoxic	X_I
# 46	Anoxic	X_PAO
# 47	Anoxic	X_PHA
# 48	Anoxic	X_PP
# 49	Anoxic	X_S
# 50	Anoxic	X_TSS
# 51	Aerobic	S_A
# 52	Aerobic	S_ALK
# 53	Aerobic	S_F
# 54	Aerobic	S_I
# 55	Aerobic	S_N2
# 56	Aerobic	S_NH4
# 57	Aerobic	S_NO3
# 58	Aerobic	S_O2
# 59	Aerobic	S_PO4
# 60	Aerobic	X_AUT
# 61	Aerobic	X_H
# 62	Aerobic	X_I
# 63	Aerobic	X_PAO
# 64	Aerobic	X_PHA
# 65	Aerobic	X_PP
# 66	Aerobic	X_S
# 67	Aerobic	X_TSS
# 68	Clarifier	S_A
# 69	Clarifier	S_ALK
# 70	Clarifier	S_F
# 71	Clarifier	S_I
# 72	Clarifier	S_N2
# 73	Clarifier	S_NH4
# 74	Clarifier	S_NO3
# 75	Clarifier	S_O2
# 76	Clarifier	S_PO4
# 77	Clarifier	X_AUT
# 78	Clarifier	X_H
# 79	Clarifier	X_I
# 80	Clarifier	X_PAO
# 81	Clarifier	X_PHA
# 82	Clarifier	X_PP
# 83	Clarifier	X_S
# 84	Clarifier	X_TSS

class EnvAsm2d():
    def __init__(self):
        self.t = None
        self.state = None
        self.default_inflow = np.array(get_default_inflow())
        self.dt = 15/60/24 # 15min-sampling
        self.dv_generator = None

        self.default_Qin = 24
        self.default_Qinter = 1.0 # as rate per Qin
        self.default_Qsludge = 0.2 # as rate per Qin
        self.default_SRT = 12
        self.default_DO = 3

        self.Ny = 2
        self.Nu = 5

        self.threshold = 4

        f = lambda _, state, params: np.array(get_dXdt(*state, *params))
# state as numpy.darray
# params as tuple
        self.ode_solver = ode(f)

    def init(self):
        self.t = 0
        self.state = np.array(get_initial_condtion()) # (85,)
        self.ode_solver.set_initial_value(self.state, 0)
        self.dv_generator = DV_generator(self.dt)
        self.dv_generator.init()

    def observe(self):
        S_NH4   = self.state[56]
        S_A     = self.state[68] # COD
        S_F     = self.state[70] # COD
        S_I     = self.state[71] # COD
        COD = S_A + S_F + S_I

        t = self.t 
        y = np.array((S_NH4, COD)).astype(np.float32).reshape(1,-1) # (1,2)
        return t, y

    def step(self, u):
# u: (1, Nu)
        log_threshold = np.log(self.threshold)
        maxminout = lambda u: np.max((np.min((u, log_threshold)), 
            -log_threshold))
        Qin     = np.exp(maxminout(u[0,0])) * self.default_Qin
        Qinter  = Qin * np.exp(maxminout(u[0,1])) * self.default_Qinter
        Qsludge = Qin * np.exp(maxminout(u[0,2])) * self.default_Qsludge
        SRT     = np.exp(maxminout(u[0,3])) * self.default_SRT
        DO      = np.exp(maxminout(u[0,4])) * self.default_DO

        inflow = self.default_inflow * np.exp(self.dv_generator.state[0])

        params = (Qin, Qsludge, Qinter, *inflow, SRT, DO)
        self.ode_solver.set_f_params(params)

        self.ode_solver.integrate(self.ode_solver.t+self.dt)

        self.t = self.ode_solver.t
        self.state = self.ode_solver.y # (85,)

        self.dv_generator.step()
        

class DV_generator():
    def __init__(self, dt, seed = 1):
# dt [day]
        self.state = None
        self.alpha = 1/(7/dt) # 1/7[day]
        self.omg = 2*np.pi/(12/24/dt) # 2pi/12 [rad/hr]
        self.beta = np.sqrt(1-np.exp(-2*self.alpha))
        self.randomstate = np.random.RandomState(seed = seed)

    def init(self):
        self.state = self.randomstate.randn(2)/np.sqrt(2) # (Ndv,)

    def step(self):
        w = self.randomstate.randn()
        c = np.cos(self.omg)
        s = np.sin(self.omg)
        a = np.exp(-self.alpha)
        state_next = np.array([
            a * (c * self.state[0] + s * self.state[1]) + self.beta * w,
            a * (-s * self.state[0] + c * self.state[1]) ])
        self.state = state_next


class TestCase(unittest.TestCase):
    def test_001(self):
        env = EnvAsm2d()
        env.init()
        _, y = env.observe() # (1, Ny)
        assert y.shape == (1, env.Ny) # Ny = 2
        assert y.dtype == np.float32

        u = np.random.randn(1,5)
        env.step(u)
        assert env.state.shape == (85,)
        assert np.all(~np.isnan(env.state))

        _, y = env.observe() # (1, Ny)
        assert y.shape == (1, env.Ny) # Ny = 2
        assert y.dtype == np.float32

    def test_002(self):
        env = EnvAsm2d()
        env.init()
        import time
        import sys
        t0 = time.time()
        N = 2**7
        for k1 in range(N):
            sys.stdout.write('\r%04d/%04d' % (k1, N))
            u = np.random.randn(1, 5)
            env.step(u)
            assert env.state.shape == (85,)
            assert np.all(~np.isnan(env.state))
            _, y = env.observe()
        t1 = time.time()
        print('\n')
        print((t1-t0)/N)

    def test_003(self):
        import matplotlib.pylab as plt
        dt = 15/60/24 # 15min-sampling
        gen = DV_generator(dt)
        gen.init()
        Y = []
        Y.append(gen.state[0])
        for k1 in range(96 * 21): # 3w
            gen.step()
            Y.append(gen.state[0])
        #print(np.std(Y))
        plt.plot(np.array(Y))
        plt.show()

if __name__ == "__main__":
    unittest.main()
