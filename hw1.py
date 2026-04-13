import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import math
import matplotlib.pyplot as plt


def ode_rhs(t, y, Cd):
    """ODE system for a cannonball with drag
    Arguments:
        t: time (s)
        y: state vector [x, z, vx, vz] (m, m, m/s, m/s)
        Cd: drag coefficient (kg/m)
    
    Returns:
        dy: time derivative of the state vector [vx, vz, ax, az] (m/s, m/s, m/s^2, m/s^2)

    """
    x, z, vx, vz = y
    vabs = np.sqrt(vx**2 + vz**2)

    g = 9.81 # m/s^2
    m = 5.44 # kg

    dxdt = vx
    dzdt = vz
    ax = -(Cd/m)*vabs*vx
    az = -g - (Cd/m)*vabs*vz

    return [dxdt, dzdt, ax, az] # --> solution [x, z, vx, vz]

def hit_the_ground(t, y, Cd): # same arguments that the ode_rhs function
    return y[1] # z

def simulate_experiment(Cd, angle, v0, tmax):
    """Simulate an experiment.
    
    Arguments:
        Cd: drag coefficient (kg/m)
        angle: launch angle (radians)
        v0: initial speed (m/s)
        tmax: maximum time (s)
        plot: bool, if true, plot simulation results and data
        tdata: measured impact time (s) [only used for plotting]
        xdata: measured impact location (m) [only used for plotting]
        vdata: inferred impect velocity (m/s) [only used for plotting]
    
    Returns:
        tfinal: final time (s)
        xfinal: final x position (m)
        vfinal: final speed (m/s)

    solve_ivp(fun, t_span, y0)
    """
    
    vx0 = v0*math.cos(angle)
    vz0 = v0*math.sin(angle)
    y0 = [0,0,vx0,vz0]
    t_pts = np.linspace(0, tmax, 1000)
    

    results = solve_ivp(ode_rhs,(0,tmax),y0, args=(Cd,), t_eval = t_pts, events=hit_the_ground)

    return results

# Add your solution here


def hit_the_ground(t, y, Cd): # same arguments that the ode_rhs function
    return y[1] # z
# the value is zero when the event of interest occurs (when it is True)
hit_the_ground.terminal = True
hit_the_ground.direction = -1

Cd = 0.003
angle = (36)*math.pi/180

res = simulate_experiment(Cd, angle, 100, 15)
#print(res.t_events)
#print(res)

plt.plot(res.y[0,:], res.y[1,:])
plt.ylabel('z')
plt.xlabel('x')
plt.show()