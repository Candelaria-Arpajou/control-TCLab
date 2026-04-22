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
def _(np):
    params = {'Ea': 72750,  # activation energy J/gmol
              'R': 8.314,  # gas constant J/gmol/K
                'k0': 7.2e10,  # Arrhenius rate constant 1/min
                'V': 100.0,  # Volume [L]
                'rho': 1000.0,  # Density [g/L]
                'Cp': 0.239,  # Heat capacity [J/g/K]
                'deltaH': -5.0e4,  # Enthalpy of reaction [J/mol]
                'UA':5.0e4,  # Heat transfer [J/min/K]
                'q':100.0, # Flowrate [L/min]
                'cAf':1.0,  # Inlet feed concentration [mol/L]
                'Tf':350.0,  # Inlet feed temperature [K]
                'Tc':300} # Temperature of the cooling water jacket   
    t_final = 10.0
    t_eval = np.linspace(0, t_final, 1001)

    cA0 = 0.5
    T0 = 350
    IC = [cA0, T0] # 
    return IC, params, t_eval


@app.cell
def _(IC, np, params, solve_ivp, t_eval):
    def model(t_eval, IC, params):
        Ea, R, k0, V, rho, Cp, deltaH, UA, q, cAf, Tf, Tc = params.values()

        def k(T):
            return k0 * np.exp(-Ea / (R * T))

        def eq(t, y):

            cA, T = y
            dcAdt = (q / V) * (cAf - cA) - k(T) * cA
            dTdt = (
                (q / V) * (Tf - T)
                + (-deltaH / rho / Cp) * k(T) * cA
                + (UA / V / rho / Cp) * (Tc - T)
            )

            return [dcAdt, dTdt]

        sol = solve_ivp(eq, [min(t_eval), max(t_eval)], IC, t_eval=t_eval)

        return sol


    solution = model(t_eval, IC, params)
    cA = solution.y[0]
    T = solution.y[1]
    return T, cA, model, solution


@app.cell
def _(params, plt, solution):
    def plot(sol, params = None, TcLabel = False):
        fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(10, 5))

        ax[0].plot(sol.t, sol.y[0], color="black", label="Concentration[mol/l")
        ax[0].set_ylabel("Concentration[mol/L]")
        ax[0].grid(True)
        ax[0].legend()

        if TcLabel and params is not None:
            ax[1].plot(sol.t, sol.y[1], color="orange", label= f"Tc[K] =  {params['Tc']}")
        else:
            ax[1].plot(sol.t, sol.y[1], color="orange", label= "Temperature[K]")

        ax[1].set_ylim(sol.y[1].min() - 10, sol.y[1].max() + 10)
        ax[1].legend()
        ax[1].grid(True)

        plt.show()

    plot(solution, params = params, TcLabel = True)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Effect of cooling temperature
    """)
    return


@app.cell
def _(IC, model, params, plt, t_eval):
    Tc_list = [290, 295, 300, 305, 310]
    fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(10, 5))

    for tc in Tc_list:
        params['Tc'] = tc

        new_sol = model(t_eval, IC, params)
        cA_ = new_sol.y[0]
        T_ = new_sol.y[1]

        ax[0].plot(new_sol.t, cA_, label= f"Tc[K] =  {params['Tc']}")
        ax[1].plot(new_sol.t, T_, label= f"Tc[K] =  {params['Tc']}")

    ax[0].set_ylabel("Concentration[mol/L]")
    ax[0].set_xlim(0, 13)
    ax[0].grid(True)
    ax[0].legend()

    ax[1].set_ylabel("Temperature[K]")
    ax[1].set_ylim(315 - 10, 450)
    ax[1].set_xlim(0, 13)
    ax[1].legend()
    ax[1].grid(True)

    plt.show()
    return Tc_list, new_sol


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Nullclines
    """)
    return


@app.cell
def _(IC, T, cA, np, params, plt):
    def steady_state(T, IC, params):
        Ea, R, k0, V, rho, Cp, deltaH, UA, q, cAf, Tf, Tc = params.values()
        cA0 = IC[0]
        T0 = IC[1]

        def k(T):
            return k0 * np.exp(-Ea / (R * T))

        cAsscA = (q/V)*cAf/((q/V) + k(T))
        cAssT = (rho*q*Cp*(Tf - T) + UA*(Tc - T))/(V*deltaH*k(T))

        return [cAsscA,cAssT]

    Temp = np.linspace(300.0, 500.0, 1000)
    sstate = steady_state(Temp, IC, params)

    plt.plot(Temp,sstate[0], color = "green", linestyle = '--', label = 'dcAdt = 0')
    plt.plot(Temp,sstate[1], color = "red", linestyle = '--', label = 'dTdt = 0')
    plt.scatter(IC[1],IC[0], color = 'blue', s = 50)
    plt.plot(T,cA)

    plt.ylim(0,1)
    plt.grid(True)
    plt.legend()
    plt.ylabel('Concentration[mol/L')
    plt.xlabel('Temperature[K]')

    plt.show()
    return Temp, steady_state


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Activities
    """)
    return


@app.cell
def _(
    IC,
    T,
    Tc_list,
    Temp,
    cA,
    model,
    new_sol,
    params,
    plt,
    steady_state,
    t_eval,
):
    for tc_ in Tc_list:
        params['Tc'] = tc_

        new_sol_ = model(t_eval, IC, params)
        cA__ = new_sol_.y[0]
        T__ = new_sol.y[1]

        newss = steady_state(Temp, IC, params)

        fig_ = plt.figure(figsize=(10, 4))
        gs = fig_.add_gridspec(2, 2)
    
        # Left plot (spans 2 rows)
        ax1 = fig_.add_subplot(gs[:, 0])
    
        # Right column (2 plots)
        ax2 = fig_.add_subplot(gs[0, 1])
        ax3 = fig_.add_subplot(gs[1, 1])
    
        # Example plots
        ax1.plot(Temp,newss[0], color = "green", linestyle = '--', label = 'dcAdt = 0')
        ax1.plot(Temp,newss[1], color = "red", linestyle = '--', label = 'dTdt = 0')
        ax1.scatter(IC[1],IC[0], color = 'blue', s = 50)
        ax1.plot(T,cA)
        ax1.set_ylim(0,1)
        ax1.set_ylabel('Concentration[mol/L')
        ax1.set_xlabel('Temperature[K]')
        ax1.grid(True)
        #plt.legend()
    
        ax2.plot(t_eval, cA__)
        ax2.set_ylabel('Concentration[mol/L')
        ax2.set_xlabel('Time')
        ax2.grid(True)
    
        ax3.plot(t_eval,T__)
        ax3.set_ylabel('Concentration[mol/L')
        ax3.set_xlabel('Time')
        ax3.grid(True)
    
        plt.tight_layout()
        plt.show()
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
