import marimo

__generated_with = "0.23.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    from scipy.signal import StateSpace, lsim, bode 
    import numpy as np
    import matplotlib.pyplot as plt

    return StateSpace, lsim, mo, np, plt


@app.cell
def _(StateSpace, np):
    # Cosntant definition
    car_mass = 4*450 #kg
    driver_mass = 80 #kg
    k = 1E5 #N/m
    c = 2E3 #Nm/s
    m = car_mass/4 # kg
    g = 9.81 # m/s^2

    # State space model definition
    A = np.array([[-c/m, -k/m],
                  [1,0]])
    B = np.array([[1/m],[0]])
    C = np.array([[0,1]])
    D = np.array([[0]])

    sys = StateSpace(A,B,C,D)
    print(sys)
    return driver_mass, g, sys


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Next, let’s simulate the step response of a driver getting into the car. We will use the lsim function from scipy.signal which takes as arguments:

    sys which is a LTI system, e.g., created using StateSpace

    T which is the time point, which must be evenly spaced. We use numpy.linspace to create this time grid.

    U which is the control grid defined for the time grid T. By default, lsim with linearly interpolate U. Another option is zero-order hold, which means a piecewise constant signal for U, i.e., a sequence of steps.

    X0 which are the initial conditions for the states.
    """)
    return


@app.cell
def _(driver_mass, g, lsim, np, plt, sys):
    x0 = [0,0]

    t = np.linspace(0,10,1000)

    # Calculate the input signal on the time grid
    u = (-driver_mass*g/4)*np.ones(len(t))

    _fig, _ax = plt.subplots(nrows = 2, ncols = 1, figsize=(10,5))
    _ax[0].plot(t,u)
    _ax[0].set_ylabel('Force (N)',fontsize=12)
    _ax[0].set_title('Driver getting in',fontsize=12)

    t_, yout,xout = lsim(sys,U=u, T=t, X0=x0)
    _ax[1].plot(t_,yout*100)
    _ax[1].set_xlabel('Time (s)',fontsize=12)
    _ax[1].set_ylabel('Displacement (cm)',fontsize=12)

    plt.show()
    return (x0,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Frequency response - Simulate Driving Over Speed Bumps
    """)
    return


@app.cell
def _(lsim, np, plt, sys, x0):
    def input_u(t,Amp=100, omega=4):
            return Amp*np.sin(omega*t)

    def simulate_bumps(t, u):
        ''' Simulate driving over bumps

        Arguments:
            A: amplitude (N)
            omega: frequency (rad/s)
            phi: phase angle (rad)

        Returns:
            Nothing

        Action:
            Creates plot
        '''

        tsim, y_out,x_out = lsim(sys,U=u, T=t, X0=x0)

        return tsim, y_out,x_out,u

    newt = np.linspace(0,5,1000)
    newu = input_u(newt)
    res = simulate_bumps(newt,newu)

    _fig, _ax1 = plt.subplots()

    _ax1.plot(res[0],res[1]*100, color = 'blue')
    _ax1.set_xlabel('Time (s)',fontsize=12)
    _ax1.set_ylabel('Displacement (cm)',fontsize=12)

    _ax2 = _ax1.twinx()
    _ax2.plot(res[0],res[3], color = "red")
    _ax2.set_xlabel('Time (s)',fontsize=12)
    _ax2.set_ylabel('Froce (N)',fontsize=12)


    plt.show()

    return input_u, newt, simulate_bumps


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Simulate Driving Over a Rumble Strip
    """)
    return


@app.cell
def _(input_u, newt, plt, simulate_bumps):
    _newu = input_u(newt, Amp=30, omega=20)
    _res = simulate_bumps(newt,_newu)

    _fig, _ax1 = plt.subplots()

    _ax1.plot(_res[0],_res[1]*100, color = 'blue')
    _ax1.set_xlabel('Time (s)',fontsize=12)
    _ax1.set_ylabel('Displacement (cm)',fontsize=12)

    _ax2 = _ax1.twinx()
    _ax2.plot(_res[0],_res[3], color = "red")
    _ax2.set_xlabel('Time (s)',fontsize=12)
    _ax2.set_ylabel('Froce (N)',fontsize=12)


    plt.show()
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
