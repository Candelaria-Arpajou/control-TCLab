import marimo

__generated_with = "0.23.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    from scipy.integrate import solve_ivp
    import matplotlib.pyplot as plt

    return mo, np, plt, solve_ivp


@app.cell
def _(np, solve_ivp):
    k = 100000 # N/m
    c = 2000 # Ns/m
    m = 450 # kg
    driver_m = 80 # kg
    g = 9.81

    params = [k,c,m]
    force_value = -driver_m*g/4

    t_span = [0, 5]
    t_eval = np.linspace(*t_span, 1000)

    def two_order_eq(t,y,params,f=lambda t:0.0): # Default arguments first and then non-default arguments 
        k,c,m = params
        x,v = y

        dvdt = (-k*x - c*v + f(t))/m
        dxdt = v

        return [dxdt,dvdt]

    def vector_not(t,x_,params,f=lambda t:0.0):
        k,c,m = params

        A = np.array([[0,1],[-k/m,-c/m]])
        B = np.array([[0,1/m]]).T

        x,v = x_

        dx_ = A @ x_ + B @ np.array([f(t)]) # --- f(t) must be an array, not a function type
        return dx_

    def simulation(t_span,params,force_value):
        initC = [0.0,0.0] # x = à & v = 0

        force_value = force_value

        f = lambda t: force_value # force constante appliquée dès t = 0

        # driver_gets_in = lambda t, y:two_order_eq(t,y,params, f)
        driver_gets_in = lambda t, y:vector_not(t,y,params, f) 
        sol = solve_ivp(driver_gets_in,[t_span[0],t_span[-1]],initC,t_eval = t_eval)

        return sol

    sol = simulation(t_span,params,force_value)
    x = sol.y[0]
    v = sol.y[1]
    return c, driver_m, force_value, g, k, m, simulation, sol, t_span, v, x


@app.cell
def _(plt, sol, v, x):
    fig, ax = plt.subplots(nrows = 2, ncols = 1, figsize=(10,5))
    ax[0].plot(sol.t, x*100 ,color='b')
    #ax[0].set_xlabel('Time (s)',fontsize=12)
    ax[0].set_ylabel('Displacement (cm)',fontsize=12)

    ax[1].plot(sol.t, v*100,color='r')
    ax[1].set_xlabel('Time (s)',fontsize=12)
    ax[1].set_ylabel('Velocity (cm/s)',fontsize=12)
    ax[1].axhline(y = 0, linewidth = 0.5, color = 'black', linestyle='--')
    ax[1].set_ylim(-2.7,1.7)
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Activities
    """)
    return


@app.cell
def _(c, force_value, m, plt, simulation, t_span):
    # Double spring stiffness.

    k_double = 100000 * 2
    _solu = simulation(t_span,[k_double,c,m], force_value)
    _x_solu = _solu.y[0]
    _v_solu = _solu.y[1]

    _fig, _ax = plt.subplots(nrows = 2, ncols = 1, figsize=(10,5))
    _ax[0].plot(_solu.t, _x_solu*100 ,color='b')
    #ax[0].set_xlabel('Time (s)',fontsize=12)
    _ax[0].set_ylabel('Displacement (cm)',fontsize=12)

    _ax[1].plot(_solu.t,_v_solu*100,color='r')
    _ax[1].set_xlabel('Time (s)',fontsize=12)
    _ax[1].set_ylabel('Velocity (cm/s)',fontsize=12)
    _ax[1].axhline(y = 0, linewidth = 0.5, color = 'black', linestyle='--')
    plt.show()
    return


@app.cell
def _(force_value, k, m, plt, simulation, t_span):
    # Double strength of the dampers.
    c_double = 2000 * 2
    _solu = simulation(t_span,[k,c_double,m], force_value)
    _x_solu = _solu.y[0]
    _v_solu = _solu.y[1]

    _fig, _ax = plt.subplots(nrows = 2, ncols = 1, figsize=(10,5))
    _ax[0].plot(_solu.t, _x_solu*100 ,color='b')
    #ax[0].set_xlabel('Time (s)',fontsize=12)
    _ax[0].set_ylabel('Displacement (cm)',fontsize=12)

    _ax[1].plot(_solu.t,_v_solu*100,color='r')
    _ax[1].set_xlabel('Time (s)',fontsize=12)
    _ax[1].set_ylabel('Velocity (cm/s)',fontsize=12)
    _ax[1].axhline(y = 0, linewidth = 0.5, color = 'black', linestyle='--')
    plt.show()
    return


@app.cell
def _(c, driver_m, g, k, m, plt, simulation, t_span):
    #Four people instead of one person gets in the car

    new_force_value = -driver_m*g/4 * 4
    _solu = simulation(t_span,[k,c,m], new_force_value)
    _x_solu = _solu.y[0]
    _v_solu = _solu.y[1]

    _fig, _ax = plt.subplots(nrows = 2, ncols = 1, figsize=(10,5))
    _ax[0].plot(_solu.t, _x_solu*100 ,color='b')
    #ax[0].set_xlabel('Time (s)',fontsize=12)
    _ax[0].set_ylabel('Displacement (cm)',fontsize=12)

    _ax[1].plot(_solu.t,_v_solu*100,color='r')
    _ax[1].set_xlabel('Time (s)',fontsize=12)
    _ax[1].set_ylabel('Velocity (cm/s)',fontsize=12)
    _ax[1].axhline(y = 0, linewidth = 0.5, color = 'black', linestyle='--')
    plt.show()
    return


@app.cell
def _(np, plt):
    # Comparison of Modes
    # Overdamped, critically dampeded and underdamped

    def second_order(t,xbar=1.0, zeta=1.0, omega=10.0):
        ''' Analytic solution of second order system

            Arguments:
                t: times to evaluate solution
                xbar: steady-state after step response
                zeta: damping coefficient
                omega: natural frequency

            Note: t and omega need to have consistent units (time and inverse time)

            Returns:
                x(t), states x evaluated at times t
        '''
        if zeta > 1:
            return xbar*(1-np.exp(-zeta*t*omega)*(np.cosh(t*omega*np.sqrt(zeta**2-1)) + (zeta/np.sqrt(zeta**2-1))*np.sinh(t*omega*np.sqrt(zeta**2 - 1))))
        elif np.abs(zeta - 1) < 1E-6:
            return xbar*(1-(1+t*omega)*np.exp(-t*omega))
        else:
            return xbar*(1-np.exp(-zeta*t*omega)*(np.cos(t*omega*np.sqrt(1-zeta**2)) + (zeta/np.sqrt(1-zeta**2))*np.sin(t*omega*np.sqrt(1-zeta**2))))

    teval = np.linspace(0,5,1001)
    x_over = second_order(teval,zeta=1.5)
    x_crit = second_order(teval)
    x_under = second_order(teval,zeta=0.5)


    plt.plot(teval,x_under,color="blue",linestyle="-",label="Underdamped ($\zeta = 0.5$)",linewidth=2)
    plt.plot(teval,x_crit, color="green",linestyle="--",label="Critcally damped ($\zeta = 1.0$)",linewidth=2)
    plt.plot(teval,x_over,color="red",linestyle="-.",label="Overdamped ($\zeta=1.5$)",linewidth=2)
    plt.xlabel("Time, $t$",fontsize=12)
    plt.ylabel("State, $x(t)$",fontsize=12)
    plt.legend(fontsize=12)
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Revised version
    """)
    return


@app.cell
def _(np, solve_ivp):
    def suspension(t,y, k, c, m, f=lambda t: 0.0):
        '''Linear model of a car suspension system

        Arguments:
        t: time (s)
        y: array of the state variables [x, v] 
            where x is the displacement (m) and v is the velocity (m/s)
        f: extenal forcing function (N)
        k: spring constant (N/m)
        c: damping constant (Ns/m)
        m: mass per wheel (kg)

        On place donc :
        - variables dynamiques
        - excitation externe
        - paramètres du modèle
        '''
        x,v = y

        vdot = (-k*x - c*v + f(t))/m
        xdot = v

        return np.array([xdot,vdot])

    def simulate_driver_getting_in(driver_mass,k,c,m):
        ''' Simulate the suspension system with a step input (driver getting in)

        Arguments:
        driver_mass: mass of the driver (kg)
        k: spring constant (N/m)
        c: damping constant (Ns/m)
        '''
        initial_conditions = [0.0,0.0]
        t_span = [0,5]
        t_eval = np.linspace(*t_span,1000) #* used to unpacked the elements of t_span
        g = 9.81 # m/s^2

        driver_gets_in = lambda t,y: suspension(t,y,k,c,m,lambda t_: -driver_mass*g/4)
        solution = solve_ivp(driver_gets_in, t_span,initial_conditions, t_eval=t_eval)

        _omega_n = np.sqrt(k/m)
        _zeta = c/(2*m*_omega_n)

        print("omega_n =",round(_omega_n,2), "rad/s")
        print("zeta =",round(_zeta,2), "dimensionless")

        return solution

    return (simulate_driver_getting_in,)


@app.cell
def _(plt, simulate_driver_getting_in):
    sol1 = simulate_driver_getting_in(driver_mass=80,k=1E5,c=1E3,m=450)
    sol2 = simulate_driver_getting_in(driver_mass=80,k=1E5,c=1E5,m=450)

    plt.figure()

    plt.plot(sol1.t, sol1.y[0]*100, label="sol1")
    plt.plot(sol2.t, sol2.y[0]*100, label="sol2")

    plt.xlabel("Time (s)", fontsize=18)
    plt.ylabel("Displacement (cm)", fontsize=18)
    plt.legend()
    plt.grid()
    plt.show()

    plt.figure()

    plt.plot(sol1.t, sol1.y[1]*100, label="sol1")
    plt.plot(sol2.t, sol2.y[1]*100, label="sol2")

    plt.xlabel("Time (s)", fontsize=18)
    plt.ylabel("Velocity (cm/s)", fontsize=18)
    plt.legend()
    plt.grid()
    plt.show()
    return


@app.cell
def _(np):
    def car_revisited(car_mass=4*450, driver_mass=80, k=1E5, c=2E3, verbose=True):
        ''' Convert car suspension example into 2nd order system general form

        Arguments:
            car_mass: total mass of the vehicle without driver (kg)
            driver_mass: total mass of the driver (kg)
            k: spring constant (N/m)
            c: damping constant (Ns/m)

        Returns:
            omega_n: natural frequency (1/s)
            zeta: critical coefficient (dimensionless)

        Notes:
            All masses are divided by 4 within the calculations to convert from total to per wheel.
            User should input total masses.

            This function returns omega and zeta for the loaded car
        '''
        if verbose:
            print("Spring constant, k =",round(k,0),"N/m")
            print("Damping constant, c =",round(c,0),"Ns/m")

        m = car_mass/4
        omega_n = np.sqrt(k/m)
        zeta = c/(2*m*omega_n)

        if verbose:
            print("\nEmpty car, m =", round(m,1),"kg per wheel")
            print("omega_n =",round(omega_n,2), "rad/s")
            print("zeta =",round(zeta,2), "dimensionless")

        m = (car_mass + driver_mass)/4
        omega_n = np.sqrt(k/m)
        zeta = c/(2*m*omega_n)

        if verbose:
            print("\nLoaded car, m =", round(m,1),"kg per wheel")
            print("omega_n =",round(omega_n,2), "rad/s")
            print("zeta =",round(zeta,2), "dimensionless")

        return omega_n, zeta

    res = car_revisited()
    print(res)
    return (car_revisited,)


@app.cell
def _(car_revisited, np, plt):
    k_adjust = np.linspace(1E3,1E4,101)

    _omega, _zeta = car_revisited(car_mass=4*450, driver_mass=80, k=k_adjust, c=2E3, verbose=False)

    _fig, _ax1 = plt.subplots()

    _ax1.set_title('Tuning Spring Constant',fontsize=15)
    _ax1.plot(k_adjust,_omega, color = 'blue')
    _ax1.set_ylabel('$\omega_n$ (rad/s)',fontsize=12)
    _ax1.set_xlabel('$k$ (N / s)',fontsize=12)

    _ax2 = _ax1.twinx()
    _ax2.plot(k_adjust,_zeta,color='red')
    _ax2.set_ylabel('$\zeta$ (dimensionless)', fontsize=12)
    plt.show()
    return


@app.cell
def _(car_revisited, np, plt):
    c_adjust = np.linspace(1E4,3E4,21)

    _omega, _zeta = car_revisited(c=c_adjust,verbose=False)

    # Convert from scalar to vector
    _omega = _omega*np.ones(len(c_adjust))

    _fig, _ax1 = plt.subplots()

    _ax1.set_title('Tuning Spring Constant',fontsize=15)
    _ax1.plot(c_adjust,_omega, color = 'blue')
    _ax1.set_ylabel('$\omega_n$ (rad/s)',fontsize=12)
    _ax1.set_xlabel('$c$ (N m / s)',fontsize=12)

    _ax2 = _ax1.twinx()
    _ax2.plot(c_adjust,_zeta,color='red')
    _ax2.set_ylabel('$\zeta$ (dimensionless)', fontsize=12)
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Sensitivity of Eigenvalues
    """)
    return


@app.cell
def _():
    import sympy

    c_, m_, k_ = sympy.symbols('c m k')
    A = sympy.Matrix([[-c_/m_, -k_/m_],[1, 0]])
    print("A = \n", A)
    print("Eigenvalues(A) = ")
    print(A.eigenvals())
    return


@app.cell
def _(np):
    def eigenvalues_car_revisited(car_mass = 4*450, driver_mass=80, k=1E5, c=2E3, verbose=True):
        ''' Convert car suspension example into 2nd order system general form

        Arguments:
            car_mass: total mass of the vehicle without driver (kg)
            driver_mass: total mass of the driver (kg)
            k: spring constant (N/m)
            c: damping constant (Ns/m)

        Returns:
            eigenvalues

        Notes:
            All masses are divided by 4 within the calculations to convert from total to per wheel.
            User should input total masses.

            This function assumes k and c are scalars!

        '''
    
        if verbose:
            print("Spring constant, k =",round(k,0),"N/m")
            print("Damping constant, c =",round(c,0),"Ns/m")

        m = car_mass/4
        omega_n = np.sqrt(k/m)
        zeta = np.sqrt(c/(2*omega_n*m))
        A = np.array([[-c/m, -k/m],[1.,0]])
        evals, evec = np.linalg.eig(A)
    
        if verbose:
            print("\nEmpty car, m =", round(m,1),"kg per wheel")
            print("omega_n =",round(omega_n,2), "1/s")
            print("zeta =",round(zeta,2), "dimensionless")
            print("eignvalues=",evals)

        m = (car_mass + driver_mass)/4
        omega_n = np.sqrt(k/m)
        zeta = np.sqrt(c/(2*omega_n*m))
        A = np.array([[-c/m, -k/m],[1.,0]])
        evals, evec = np.linalg.eig(A)

        if verbose:
            print("\nLoaded car, m =", round(m,1),"kg per wheel")
            print("omega_n =",round(omega_n,2), "1/s")
            print("zeta =",round(zeta,2), "dimensionless")
            print("eignvalues=",evals)

        return evals

    results = eigenvalues_car_revisited()
    print(results)
    return (eigenvalues_car_revisited,)


@app.cell
def _(eigenvalues_car_revisited, np, plt):
    # Perform sensitivity analysis of eigenvalues to k
    _k_adjust = np.linspace(1E3,1E4,101)

    evals = np.zeros((len(_k_adjust),2),dtype = complex)


    for i in range(len(_k_adjust)):
        evals[i,:] = eigenvalues_car_revisited(k=_k_adjust[i],verbose=False)

    _fig, ax1 = plt.subplots()

    ax1.plot(_k_adjust/1E3,np.real(evals[:,0]),label="real($\lambda_1$)",color="blue",linestyle="-",linewidth=2)
    ax1.plot(_k_adjust/1E3,np.real(evals[:,1]),label="real($\lambda_2$)",color="blue",linestyle="--",linewidth=2)
    ax1.set_ylabel('Real', color='blue',fontsize=18)
    ax1.set_xlabel('$k$ (kN / s)',fontsize=18)
    ax1.tick_params(axis='y', color='blue', labelcolor='blue')
    ax1.set_title('Tuning Spring Constant',fontsize=18)
    plt.legend(loc='upper left')
    ax2 = ax1.twinx()
    ax2.plot(_k_adjust/1E3,np.imag(evals[:,0]),label="imag($\lambda_1$)",color='red',linestyle="-",linewidth=2)
    ax2.plot(_k_adjust/1E3,np.imag(evals[:,1]),label="imag($\lambda_2$)",color='red',linestyle="--",linewidth=2)
    ax2.set_ylabel('Imaginary', color='red',fontsize=18)
    ax2.tick_params(axis='y', color='red', labelcolor='red')
    ax2.spines['right'].set_color('red')
    ax2.spines['left'].set_color('blue')
    plt.legend(loc='upper right')
    plt.show()
    return


@app.cell
def _(eigenvalues_car_revisited, np, plt):
    # Perform sensitivity analysis of eigenvalues to c

    _c_adjust = np.linspace(1E4,3E4,201)

    _evals = np.zeros((len(_c_adjust),2),dtype = complex)

    for indx in range(len(_c_adjust)):
        _evals[indx,:] = eigenvalues_car_revisited(c=_c_adjust[indx],verbose=False)

    _fig, _ax1 = plt.subplots()

    _ax1.plot(_c_adjust/1E3,np.real(_evals[:,0]),label="real($\lambda_1$)",color="blue",linestyle="-",linewidth=2)
    _ax1.plot(_c_adjust/1E3,np.real(_evals[:,1]),label="real($\lambda_2$)",color="blue",linestyle="--",linewidth=2)
    _ax1.set_ylabel('Real', color='blue',fontsize=18)
    _ax1.set_xlabel('$c$ (kN m / s)',fontsize=18)
    _ax1.tick_params(axis='y', color='blue', labelcolor='blue')
    _ax1.set_title('Tuning Damper Constant',fontsize=18)
    plt.legend(loc='lower right')

    _ax2 = _ax1.twinx()
    _ax2.plot(_c_adjust/1E3,np.imag(_evals[:,0]),label="imag($\lambda_1$)",color='red',linestyle="-",linewidth=2)
    _ax2.plot(_c_adjust/1E3,np.imag(_evals[:,1]),label="imag($\lambda_2$)",color='red',linestyle="--",linewidth=2)
    _ax2.set_ylabel('Imaginary', color='red',fontsize=18)
    _ax2.tick_params(axis='y', color='red', labelcolor='red')
    _ax2.spines['right'].set_color('red')
    _ax2.spines['left'].set_color('blue')

    plt.legend(loc='upper right')
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Simulation with scipy.signal
    """)
    return


if __name__ == "__main__":
    app.run()
