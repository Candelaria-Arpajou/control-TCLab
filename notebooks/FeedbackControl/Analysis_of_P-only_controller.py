import marimo

__generated_with = "0.23.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy.integrate import solve_ivp

    return np, plt, solve_ivp


@app.cell
def _(np, plt, solve_ivp):
    # Parametres
    T_amb = 21
    alpha = 0.00016
    P1 = 200
    U1 = 50
    T_set = 40

    # fittes parameters for hardware
    Ua = 0.05
    Ub = 0.05
    CpH = 5.0
    CpS = 1.0

    t_final = 800
    t_step = 1
    t_expt = np.arange(0, t_final, t_step)

    def system(t,y):
        T1H, T1S = y
        dT1H = (-(Ua + Ub)*T1H + Ub*T1S + alpha*P1*U1 + U1*T_amb)/CpH
        dT1S = Ub*(T1H - T1S)/CpS

        return [dT1H, dT1S]

    sol = solve_ivp(system, [t_expt[0],t_expt[-1]], [T_amb, T_amb], t_eval=t_expt)

    plt.plot(sol.t, sol.y[0],label='T1H')
    plt.plot(sol.t, sol.y[1],label='T1S')
    plt.legend()
    plt.xlabel('Time [second]')
    plt.ylabel('Temperature [° C]')
    plt.grid()
    plt.show()
    return CpH, CpS, P1, T_amb, T_set, Ua, Ub, alpha, t_expt


@app.cell
def _(CpH, CpS, P1, T_amb, T_set, Ua, Ub, alpha, np, plt, solve_ivp, t_expt):
    def simulate_response(Kp = 0.0):
        A = np.array([[-(Ua + Ub)/CpH, (Ub - alpha*P1*Kp)/CpH],
                      [Ub/CpS, -Ub/CpS]])
        def closed_loop(t,y):
            return A @ y
        sol_P = solve_ivp(closed_loop, [t_expt[0],t_expt[-1]], [T_amb- T_set, T_amb - T_set], t_eval=t_expt)

        plt.plot(sol_P.t, sol_P.y[0] + T_set,label='T1H')
        plt.plot(sol_P.t, sol_P.y[1] + T_set,label='T1S')
        plt.axhline(T_set, color = 'black', linewidth = 0.5)
        plt.title("$K_p$ = "+str(Kp))
        plt.grid()
        plt.legend()
        plt.xlabel('Time [second]')
        plt.ylabel('Temperature [° C]')
        plt.show()

    simulate_response(Kp=3.2)
    return


@app.cell
def _(CpH, CpS, P1, Ua, Ub, alpha, np):
    from scipy.signal import cont2discrete

    def continuous_system(Kp):
        ''' Continous system for TCLab with P control

        Arguments:
            Kp: the proportional control gain

        Returns:
            A, B, C, D: the state space matrices

        '''

        A = np.array([[-(Ua + Ub)/CpH, (Ub - alpha*P1*Kp)/CpH], 
                    [Ub/CpS, -Ub/CpS]])

        B = np.array([[alpha*P1*Kp/CpH], [0]])

        C = np.array([[0, 1]])

        D = np.array([[0]])

        return A, B, C, D

    def discrete_system(Kp):
        ''' Discrete system for TCLab with P control

        Arguments:
            Kp: the proportional control gain

        Returns:
            Ad, Bd, Cd, Dd: the state space matrices

        Notes:
            The time step is assumed to be 1 second

        '''
        A, B, C, D = continuous_system(Kp)

        d_system = cont2discrete((A, B, C, D), 1, method='zoh')

        Ad = d_system[0]
        Bd = d_system[1]
        Cd = d_system[2]
        Dd = d_system[3]

        return Ad, Bd, Cd, Dd

    Ad, Bd, Cd, Dd = discrete_system(Kp=1.0)
    print("Ad=\n",Ad)
    print("Bd=\n",Bd)
    print("Cd=\n",Cd)
    print("Dd=\n",Dd)
    return (continuous_system,)


@app.cell
def _(np):
    t = np.arange(0, 300, 1)
    T_sp = np.ones(t.shape)*5
    T_sp[100:] = 30
    return T_sp, t


@app.cell
def _(T_amb, T_sp, continuous_system, plt, t):
    from scipy.signal import lsim, lti

    def simulate_response_continous(Kp=1.0):

        c_system = lti(*continuous_system(Kp))

        T, yout, xout = lsim(c_system, T_sp, t, X0=[0, 0])

        plt.plot(t, xout[:,0] + T_amb, label='T1H')
        plt.plot(t, xout[:,1] + T_amb, label='T1S')
        plt.plot(t, T_sp + T_amb, label='Setpoint', linestyle='-.', color='black', alpha=0.5)
        plt.xlabel('Time [second]')
        plt.ylabel('Temperature [° C]')
        plt.grid()
        plt.legend()
        plt.show()

    simulate_response_continous(Kp=10.0)
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
