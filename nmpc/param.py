import numpy as np
## Constantes
g   = 9.81;   # Gravitational acceleration constant [m/s�]
Cc = 2e-5 ;   # Choke valve constant
A1 = 0.008107;# Cross-section area of pipe below ESP [m�]
A2 = 0.008107;# Cross-section area of pipe above ESP [m�]
D1 = 0.1016;  # Pipe diameter below ESP [m]
D2 = 0.1016;  # Pipe diameter above ESP [m]
h1 = 200;     # Heigth from reservoir to ESP [m]
hw = 1000;    # Total vertical distance in well [m]
L1 =  500;    # Length from reservoir to ESP [m]
L2 = 1200;    # Length from ESP to choke [m]
V1 = 4.054;   # Pipe volume below ESP [m3]
V2 = 9.729;   # Pipe volume above ESP [m3]
f0 = 60;      # ESP characteristics reference freq [Hz]
q0_dt = 25/3600; # Downtrhust flow at f0
q0_ut = 50/3600; # Uptrhust flow at f0
Inp = 65;     # ESP motor nominal current [A]
Pnp = 1.625e5;# ESP motor nominal Power [W]
b1 = 1.5e9;   # Bulk modulus below ESP [Pa]
b2 = 1.5e9;   # Bulk modulus above ESP [Pa]

M  = 1.992e8; # Fluid inertia parameters [kg/m4]
rho = 950;    # Density of produced fluid [kg/m�?³]
pr = 1.26e7;  # Reservoir pressure
PI = 2.32e-9; # Well productivy index [m3/s/Pa]
mu  = 0.025;  # Viscosity [Pa*s]
dfq_max = 0.5;    # m�xima varia��o em f/s
dzc_max = 1;  # m�xima varia��o em zc #/s
tp =np.array([[1/dfq_max,1/dzc_max]]).T;  # Actuator Response time
CH = -0.03*mu + 1;
Cq = 2.7944*mu**4 - 6.8104*mu**3 + 6.0032*mu**2 - 2.6266*mu + 1;
Cp = -4.4376*mu**4 + 11.091*mu**3 -9.9306*mu**2 + 3.9042*mu + 1;

# # Alterações
# b1 = 1.5;   # Bulk modulus below ESP [Pa]
# b2 = 1.5;   # Bulk modulus above ESP [Pa]
# M  = 1.992; # Fluid inertia parameters [kg/m4]
# pr = 1.26;  # Reservoir pressure

print('Dados carregados')