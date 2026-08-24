import marimo

__generated_with = "0.23.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _():
    from tclab import TCLab, clock, Historian, Plotter, setup
    import matplotlib.pyplot as plt

    # Important notes:
    # connected=False means we use a digital twin instead of connecting to a real device
    # speedup=20 speeds up the simulation by a factor of 20
    TCLab = setup(connected=False, speedup=20)

    # control parameters
    U_min = 0
    U_max = 100
    T_SP = 40
    d = 0.5

    # time horizon and time step
    t_final = 250
    t_step = 1

    # perform experiment
    with TCLab() as lab:
        lab.P1 = 200
        h = Historian(lab.sources)
        #p = Plotter(h, t_final)
        for t in clock(t_final, t_step):
            T1 = lab.T1                             # measure temperature
            if T1 <= T_SP - d:
                U1 = U_max
            elif T1 >= T_SP + d:
                U1 = U_min

            lab.Q1(U1)
            h.update(t)

            #U1 = U_max if lab.T1 < T_SP else U_min  # compute manipulated variable
            #lab.Q1(U1)                              # adjust power
            #h.update(t)                             # log results

    p = Plotter(h, t_final)
    p.update(t_final)

    plt.show()
    return (plt,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Numeric simulation
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### State space et lsim
    """)
    return


@app.cell
def _(plt):
    import numpy as np
    from scipy.signal import StateSpace, lsim

    Ua = 0.07
    Ub = 0.04
    CpH = 4
    CpS = 1
    alpha = 0.00016
    P1 = 200
    T_amb = 21

    A = np.array([[-(Ua + Ub)/CpH, Ub/CpH],[Ub/CpS,-Ub/CpS]])
    B = np.array([[alpha*P1/CpH], [0]])
    C = np.array([[0,1]])
    D = np.array([[0]])

    sys = StateSpace(A, B, C, D)
    t_ = np.linspace(0,1000,1001)

    u = np.zeros(len(t_))
    u[20:] = 100

    # Simulate de model
    t_, y, x = lsim(sys, u, t_)

    # Plot the results
    plt.plot(t_, y + T_amb)
    plt.xlabel('Time / seconds')
    plt.ylabel('Temperature / °C')
    plt.show()

    plt.plot(t_, u)
    plt.xlabel('Time / seconds')
    plt.ylabel('Power / %')
    plt.show()
    return A, B, C, D, T_amb, np


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Implementation of the Relay controller
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Histeresis relay control avec changement du SP
    """)
    return


@app.cell
def _(A, B, C, D, T_amb, np, plt):
    from scipy.signal import cont2discrete

    #IMPORTANTE: la entrada (input u) depende del valor de la salida (output y). Por esa razon no se puede definir u completamente antes de resolver el sistema. Es necesario discretizar 

    t_sample = np.linspace(0,1000,1001)

    #SP = 40
    _d = 1 # Deadband °C

    n = len(t_sample)
    _x = np.zeros((n,2))
    _y = np.zeros(n)
    _u = np.zeros(n)
    SP = np.zeros(n)
    SP[:299] = 40
    SP[300:] = 50
    print(SP)

    Qmax = 70
    Qmin = 0

    Ad, Bd, Cd, Dd, dt = cont2discrete((A,B,C,D), dt=1, method='zoh')

    for i in range(1,len(t_sample)):
        if _y[i-1] <= (SP[i-1] - T_amb) - _d:
            _u[i] = Qmax
        elif _y[i-1] >= (SP[i-1] - T_amb) + _d:
            _u[i] = Qmin
        else:
            _u[i] = _u[i-1]

        # Update the state
        _x[i,:] = Ad @ _x[i-1] + Bd.T * _u[i]
        _y[i] = (Cd @ _x[i] + Dd.T * _u[i])[0,0]

    plt.plot(t_sample, _y + T_amb)
    plt.xlabel('Time / seconds')
    plt.ylabel('Temperature / °C')
    plt.show()

    plt.plot(t_sample, _u)
    plt.xlabel('Time / seconds')
    plt.ylabel('Power / %')
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
