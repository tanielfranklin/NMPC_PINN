import numpy as np
yss=np.vstack([6000142.88550200, 592.126490003812])
q = np.vstack([1e6, 1e8]) / (yss**2)
print(q)
print(np.tile(q[:,0],2))
Q = np.diag(np.tile(q[:,0],2))
print('Q')
print(Q)
