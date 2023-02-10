import matplotlib.pyplot as plt
from scipy.optimize import minimize
import numpy as np
from scipy.integrate import odeint
from scipy.optimize import fsolve

o2_array_full = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8])
m_full = np.array([1,0.58, 0.46, 0.21, 0.12, 0.06, 0.04, 0.03, 0.012])
plt.plot(o2_array_full,m_full)

def mortality_model(params):
    km = params[0]
    mortality = np.exp(-km*o2_array_full)
    return mortality

def model_error(params): 
    mortality_hat = mortality_model(params)
    return np.sum((m_full-mortality_hat)**2)

p0 = [1]
res = minimize(model_error, p0)
print("km estimate:", res.x)

km = .48959832 # determined by regression given the supplied data

#define model and putting all units in mg
def fish_ox(y,t):
    o2 = y[0]
    fish = y[1]
    dF_dt = .05*fish - (np.exp(-km*(o2)))*fish
    dO2_dt = 16 - 1.2*(fish) -2*o2
    return [dO2_dt, dF_dt]

#simulation parameters
end = 300; dt = 1

times = np.arange(0, end+dt, dt)

#odient function to integrate
O2_0 = 8; fish_0 = 1
Y = odeint(fish_ox, [O2_0,fish_0], times)
o2_output = Y[:,0]
fish_output = Y[:,1]

# plot results for change in state variable over time time
fig, ax1 = plt.subplots()
ax1.set_xlabel('Time (days)')
ax1.set_ylabel('Fish biomass concentration (g/L)')
ax1.plot(times, fish_output, 'b', label = "fish biomass")
ax1.tick_params(axis ='y', labelcolor = 'b')

ax2 = ax1.twinx()

ax2.set_ylabel('O2 concentration (mg/L)')
ax2.plot(times, o2_output, 'r', label = "O2 concentration")
ax2.tick_params(axis ='y', labelcolor = 'r')

ax1.plot(np.nan, '-r', label = 'O2 concentration') # dummy plot to add line color to ax1 for legend formatting
ax1.legend(loc=7)

fishMax = np.max(fish_output)
o2Max = np.max(o2_output)

# Different combinations to evaluate model at
X = np.linspace(0,10,18)    
Y = np.linspace(0,5,18)
U, V = np.meshgrid(X, Y)


# List of equlibrium points
eqPoints = []

# Check all combinations
for i in range(0,len(X)):
    for j in range(0,len(Y)):
        o2 = X[i]
        fish = Y[j]
        # Calculate changes in H and P
        [dU,dV] = fish_ox([o2, fish], 0)
        U[j,i] = dU # np.log(dU)
        V[j,i] = dV
        # Find the 'roots' for this starting spot
        x = fsolve( fish_ox, [o2,fish], 0)
        eqPoints.append(x)

# Round the equlibrium points and find the unique coordinates
eqPoints = np.round(np.array(eqPoints),4)
eqPoints = np.unique(eqPoints,axis=0)
print(eqPoints)    

# Plot the Result
fig, ax = plt.subplots()
q = ax.quiver(X, Y, U, V, angles = "xy")
plt.plot(o2_output, fish_output, '-', color='green')
plt.plot(eqPoints[:,0],eqPoints[:,1],'gx',label='Equlibrium Points')
plt.xlabel("O2 concentration (mg/L)")
plt.ylabel("Fish concentration (g/L)")

plt.show()