import numpy as np
from scipy.integrate import solve_bvp, solve_ivp, simpson
from scipy.optimize import minimize_scalar, fminbound, brute, shgo

from IPython.display import Latex

from dataclasses import dataclass

method = "LSODA"
method = "Radau"
# options_solver = dict(ma)

@dataclass
class internal_bore():
    """Class for organizing parameters and functions"""
    
    H: float #actual depth
    H_0: float #depth scale to set stratification profile
    rho_s: float = 1024 #surface density
    rho_b: float = 1026 #bottom density
    rho_0: float = 1025 #reference density
    𝜆: float = 5.0
    z0: float = 0.7
    ε: float = 0
    d0: float = 0
    g: float = 9.81
    #c: float = 0.5*np.sqrt(gp*H) #two-layer conjugate state speed Lamb 2000
    
    def __post_init__(self):
        #self.zm = np.linspace(0, -self.H, 100) # ambient vertical coordinate 0 = surface
        self.za = np.linspace(0, self.H, 101) # ambient vertical coordinate (MAB)
        self.z  = self.za.copy() # displace vertical coordinate
        self.n  = np.zeros_like(self.za) # isopycnal displacement
        self.nz = np.zeros_like(self.za) # isopycnal displacement vertical gradient
        
        self.gp = self.g*(self.rho_b - self.rho_s)/self.rho_0
        self.b_s,_ = self.stratification(self.H_0, shape = self.𝜆, z0 = self.z0) # W&H 2014 eqn. 4.3
        self.b_b,_ = self.stratification(0, shape = self.𝜆, z0 = self.z0) # W&H 2014 eqn. 4.3 derivative
        #self.b_b,_ = self.stratification(self.H_0 - self.H, shape = self.𝜆, z0 = self.z0) # W&H 2014 eqn. 4.3 derivative
        
        self.b, self.bz = self.buoyancy(self.za + self.H_0 - self.H)
        self.N20 = (-self.g/self.rho_0)*self.bz[0]
        
    def stratification(self, z, shape, z0):
        val = shape*( z - self.H_0*self.z0 )/self.H_0
        #val = shape*(z - z0)
        val = np.clip(val, a_min = -355, a_max = 355) #clip to prevent overflow
        b_hat = (1/2)*(1 - np.tanh(val))
        bz_hat = -(1/2)*(shape/self.H_0)/(np.cosh(val)**2)    
        return b_hat, bz_hat
        
    def buoyancy(self, z):
        """W&H 2014 eqn. 4.2"""
        b_hat, bz_hat = self.stratification(z, shape = self.𝜆, z0 = self.z0)
        b = (b_hat - self.b_s)/(self.b_b - self.b_s)
        bz = bz_hat/(self.b_b - self.b_s)
        return b, bz

    def bvf(self):
        self.N2 = self.gp*self.bz
        return self.N2

    def density(self):
        self.rho = self.b*(self.rho_b - self.rho_s) + self.rho_s 
        return self.rho
    
    def dissipation(self, cb, delta):
        """W&H 2014 3.10"""
        #self.D = np.trapz(cb*(1 - self.nz)*delta, self.z) 
        self.D = simpson(cb*(1 - self.nz)*delta, self.z) # 
        return self.D
    
    def speed(self):
        """W&H 2014 eqn. 3.10 rearranged to solve for U"""
        self.U = self.D/(self.H*self.d0) 
        return self.U

#     def streamline_analytic(self, N, c):
#         b = (self.d0*self.ε)/(self.gp*self.rho_0)
#         w = N/c
#         n = -b*(  (1/np.sin(w*self.H) - 1/np.tan(w*self.H))*np.sin(w*self.za) + np.cos(w*self.za) - 1 )
#         return n
    
    def ode_ivp(self, z, y, Cb):
        """Solve W&H 2014 3.7 as initial value problem"""
        b, bz = self.buoyancy(z - y[0] + self.H_0 - self.H)
        N2 = self.gp*bz   
        
        if (z - y[0]) < 0:
            #display("using virtual density profile")
            N2 = self.N20*np.exp( -( (z - y[0])/ (0.01*self.H))**2 ) #W&H 3.14
        
        y1 = y[1]
        #y2 = -y[0]*N2/Cb**2 - delta_z/(rho_0*Cb**2*(y[1] - 1))
        #y2 = -(1/Cb**2)*( N2*y[0] + (self.d0*self.ε/self.rho_0)*(bz/(y[1] - 1)) )
        y2 = -(-N2/Cb**2)*( y[0] - (self.d0*self.ε)/(self.gp*self.rho_0) ) #W&H 2014 3.7
        return [y1,y2]
    
    def ode_bvp(self,z,y,p):
        Cb = p[0]
        b, bz = self.buoyancy(z - y[0] + self.H_0 - self.H)
        N2 = self.gp*bz   
                
        #virtual density profile
        inds = (z - y[0])<0
        if any(inds):
            N2[inds] = self.N20*np.exp( -( (z[inds] - y[0][inds])/ (0.01*self.H))**2 ) #W&H 3.14

        y1 = y[1]
        y2 = -(N2/Cb**2)*( y[0] - (self.d0*self.ε)/(self.gp*self.rho_0) ) #W&H 2014 3.7
        return np.vstack((y1,y2))

def delta_function(d0, b, ε):
    """W&H 2014 eqn. 3.5"""
    delta = d0*(1/2 + ε*(b - 0.5))
    return delta

def J_boremomn(d0, c, ib, disp = False):
    """W&H2014 eqn. 3.4"""
    
    delta = delta_function(d0, ib.b, ib.ε)
    m = (1/4)*ib.rho_0*c**2*ib.nz**3 - (1 - 0.5*ib.nz)*delta  

    #J = np.trapz(m, ib.z)
    J = simpson(m, ib.z)
    
    if disp:
        display(f"Δ0 = {d0}...J = {J}")
    
    return abs(J)

def J_boreheight(dn0, h_b, c, ib, disp = False):
    """W&H2014 eqn. 4.5"""
    
    #dn0 = np.exp(dn0)

    M = solve_ivp(ib.ode_ivp, [0, ib.H], [0, dn0], args = (c,), dense_output = True, method = method)
    n = M.sol(ib.za)[0]
    
    ib.z = ib.za + n
    b, bz = ib.buoyancy(ib.z + ib.H_0 - ib.H)
    
    #J = h_b - np.trapz(b, ib.z) 
    J = h_b - simpson(b, ib.z) 
    
    if disp:
        display(f"η'_0 = {dn0:0.3g}...hb={h_b:0.3g}...hb_est={np.trapz(b,ib.z):0.3g}...diff: {J:0.3g}")
    
    return abs(J)

def J_borespeed(c, ib, dn0 = 0.0, disp = False):
    """vary bore speed until boundary condition η(H) = 0 satisfied"""
    M = solve_ivp(ib.ode_ivp, [0, ib.H], [0, dn0], args = (c,), dense_output = True, method = method)
    J = M.sol(ib.H)[0]
    #J = M.y[0][-1] 
    
    if disp:
        display(f"c = {c}...J = {J}")
    
    return abs(J)

# def J_borespeed(c, ib, dn0 = 0.0):
# #     """BVP approach"""
        
#     M = solve_bvp(ib.ode_ivp, [0, ib.H], [0, dn0], args = (c,), dense_output = False, method = method)
#     #print(M.y[0][-1])
#     return abs( M.y[0][-1]  )

def bore_model(h_b, ib, cb = 0.5,
               reldiff = np.inf, tol = 0.01, maxiter = 10, disp = True, 
               output = "diss"):
    
    dn0 = 0.0
    k = 0
    while np.any(abs(reldiff) > tol):
        #print(reldiff, np.any(abs(reldiff) > 0.001))
        
        last_vals = np.array([cb, ib.d0, dn0])

        #display("calculating dissipation: varying Δ_0 to conserve momentum")
        #Find Δ
        F_m = minimize_scalar(J_boremomn, args = (cb, ib), bounds = [-1,1], method = "bounded")
        ib.d0 = F_m.x
        delta = delta_function(d0 = ib.d0, b = ib.b, ε = ib.ε)
        
        #display("calculating bore amplitude: varying η_z(0) to match h_b")
        #Find dn0
        F_h = minimize_scalar(J_boreheight, args = (h_b, cb, ib), bounds = [-1, 1], method = "bounded", options = dict(xatol = 1e-3) )
        dn0 = F_h.x
        #dn0 = np.exp(F_h.x)
        
        #display("calculating bore speed: varying C_b to match BC η(H) = 0")
        #Find Cb
        F_c = minimize_scalar(J_borespeed, args = (ib, dn0), bounds = [0, 10], method = "bounded")
        cb = F_c.x

        
        #display("solving IVP")
        #Compute solution
        M = solve_ivp(ib.ode_ivp, [0,ib.H], [0, dn0], args = (cb,), dense_output = True, method = method)
        ib.n, ib.nz = M.sol(ib.z)[0], M.sol(ib.z)[1]
        
        ib.z = ib.za + ib.n #displaced isopycnals
        ib.b, ib.bz = ib.buoyancy(ib.z + ib.H_0 - ib.H)

        new_vals = np.array([cb, ib.d0, dn0])
        reldiff =  (new_vals - last_vals)/(np.finfo(float).eps + abs(new_vals) + abs(last_vals))
        
        if disp:
            display(Latex(fr"""iter: $C_b={cb:0.2g}..η'_0={dn0:0.3g}...Δ_0={ib.d0:0.3g}$"""))

        k += 1
        if k == maxiter:
            display(f"Convergence not reached in {k} iterations")
            break
    ib.Δ = delta_function(d0 = ib.d0, b = ib.b, ε = ib.ε)
    ib.D = ib.dissipation(cb, delta)
    if (k == maxiter) or (abs(ib.D) > 1e2) or (cb == 10): 
        pass
        #ib.D = -np.inf
    
    display(f"dissipation = {ib.D:0.4g}...bore amplitude = {simpson(ib.b, ib.z):0.2g} m")
    
    if output == "ib":
        return ib
    elif output == "diss":
        return -ib.D
    
    
#Sandbox

#         display("calculating dissipation")
#         #Find Δ
#         F_m = minimize_scalar(J_boremomn, args = (ib.z, cb, ib.nz, ib.b, ib.ε, ib.rho_0))
#         ib.d0 = F_m.x
#         delta = delta_function(d0 = ib.d0, b = ib.b, ε = ib.ε)
        
#         display("calculating bore amplitude")
#         #Find dn0
#         #dn0 = brute(J_boreheight, args = (h_b, cb, ib), ranges = ((-1, 1),) )
#         result = shgo(J_boreheight, args = (h_b, cb, ib), bounds = [(-2, 2)] )
#         dn0 = result.x[0]
        
#         display("calculating bore speed")
#         #Find Cb
#         F_c = minimize_scalar(J_borespeed, args = (ib, dn0), bounds = [0, 2.5], method = "bounded")
#         cb = F_c.x
        