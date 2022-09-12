import numpy as np
from casadi import MX, Opti, Function, sin
x = MX.sym('x',2)
y = MX.sym('y')
p=MX.sym('p',3)
x=p[0:2]
y=p[-1]
f = Function('f',[p],\
           [x,sin(y)*x])
print(f)
r0, q0 = f([1.1,3.3,2])
print('r0:',r0)
print('q0:',q0)

opti = Opti()
x = opti.variable(2)
y = opti.variable()
p=opti.variable(3)
x=p[0:2]
y=p[-1]
f = Function('f',[p],\
           [x,sin(y)*x])
print(f)

r0, q0 = f([1.1,3.3,2])
print('opti r0:',r0)
print('opti q0:',q0)