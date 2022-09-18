import numpy as np
def Lim_c(xin):
  return xin[1]-xin[0]
class Parameters(object):
  def __init__(self):
    f_lim=(30,75)
    qlim_m3=[15,65]
    self.qlim=(qlim_m3[0]/3600,qlim_m3[1]/3600)
    self.F1c,self.F2c=8111.87,19468.5
    self.F1lim,self.F2lim=(99811.5, 107923),(239548,259016)
    self.Hc=1557.0851455268821
    self.qcc=0.03290348005910621
    zclim=(0,1)
    self.pmlim=(1e5,50e5)
    self.pbhlim=(1e5,1.26e7) 
    self.pwhlim=(1e5,50e5) 
    self.pbc=Lim_c(self.pbhlim)
    self.pwc=Lim_c(self.pwhlim)
    self.qc=Lim_c(self.qlim)
    self.pbmin=self.pbhlim[0]
    self.pwmin=self.pwhlim[0]
    self.qmin=self.qlim[0]
    self.H_lim=(-136.31543417849096, 1420.7697113483912)
    self.qch_lim=(0.0, 0.03290348005910621)
    # Normalizing factors
    self.prc,self.pr0=(1.4e7-1.2e7),1.2e7
    self.pm_c,self.pm0=(2.1e6-1.2e6),1.2e6
    self.xc=np.array([self.pbc,self.pwc,self.qc])
    self.x0=np.array([self.pbmin,self.pwmin,self.qmin])
    self.uc=[60,100,self.pm_c,self.prc]
    self.u0=[0,0,self.pm0,self.pr0]

  def normalizar_u(self,u):
    # u=np.array(u,ndmin=2)
    # print(u.shape)
    un=[(u[:,i]-self.u0[i])/self.uc[i] for i in range(4)]
    return np.array(un).T
  def normalizar_x(self, x):
    if x.shape[1]==3:
      xn=[(x[:,i]-self.x0[i])/self.xc[i] for i in range(3)]
    else:
      xn=[(x[:,i]-self.x0[i])/self.xc[i] for i in range(2)]
    return np.array(xn).T
  
