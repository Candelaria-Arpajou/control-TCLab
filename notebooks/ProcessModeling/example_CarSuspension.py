import marimo

__generated_with = "0.23.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    from scipy.integrate import solve_ivp
    import matplotlib.pyplot as plt

    return np, plt, solve_ivp


@app.cell
def _(np, solve_ivp):
    k = 100000 # N/m
    c = 2000 # Ns/m
    m = 450 # kg
    driver_m = 80 # kg
    g = 9.81

    params = [k,c,m]
    new_params = [k,c,-driver_m*g/4]

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

    def simulation(t_span,new_params):
        initC = [0.0,0.0] # x = à & v = 0

        force_value = new_params[-1]

        f = lambda t: force_value # force constante appliquée dès t = 0

        # driver_gets_in = lambda t, y:two_order_eq(t,y,params, f)
        driver_gets_in = lambda t, y:vector_not(t,y,params, f) 
        sol = solve_ivp(driver_gets_in,[t_span[0],t_span[-1]],initC,t_eval = t_eval)

        return sol
    
    sol = simulation(t_span,new_params)
    x = sol.y[0]
    v = sol.y[1]
    return sol, v, x


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
    plt.show()
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
