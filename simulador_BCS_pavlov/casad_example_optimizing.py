import numpy as np
import casadi as cs
import matplotlib.pyplot as plt



opti = cs.Opti()
x = opti.variable()
y = opti.variable()
p = opti.variable(2)
x = p[1]
y =p[0]


f = cs.Function('f', [p],[-2*(x-4)**2*y*(x-6)**2])

print(f(p))




exit()
opti.minimize(f(p).printme(0))


p_opts = {"expand": True}
s_opts = {"max_iter": 80, "print_level": 3}
opti.solver("ipopt", p_opts,
                    s_opts)
opti.set_initial(y,1.1)

sol = opti.subject_to(opti.bounded(0.5,x,4))

sol = opti.solve()
print(f(p), sol.value(p))

#q0 = f([1.1, 2])
#print('opti q0:', q0)
#print(sol.value(y))



y = p[-1]
f = cs.Function('f', [p],[-2*(x-4)**2*y*(x-6)**2])

x1=np.arange(0,8,0.1)
w=[]
for i in range(len(x1)):
    w.append(f([1.1,x1[i] ]))
w=np.array(w).reshape(len(x1))
plt.plot(x1,w)

print(f)
plt.show()

