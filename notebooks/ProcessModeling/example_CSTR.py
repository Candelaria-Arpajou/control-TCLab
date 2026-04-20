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
def _(np):
    # fixed model parameters
    Ea = 72750  # activation energy J/gmol
    R = 8.314  # gas constant J/gmol/K
    k0 = 7.2e10  # Arrhenius rate constant 1/min
    V = 100.0  # Volume [L]
    rho = 1000.0  # Density [g/L]
    Cp = 0.239  # Heat capacity [J/g/K]
    deltaH = -5.0e4  # Enthalpy of reaction [J/mol]
    UA = 5.0e4  # Heat transfer [J/min/K]
    q = 100.0  # Flowrate [L/min]
    cAf = 1.0  # Inlet feed concentration [mol/L]
    Tf = 350.0  # Inlet feed temperature [K]
    Tc = 300

    params = [Ea, R, k0, V, rho, Cp, deltaH, UA, q, cAf, Tf, Tc]
    t_final = 10.0
    t_eval = np.linspace(0, t_final, 1001)
    IC = [0.5, 350]
    return IC, params, t_eval


@app.cell
def _(IC, np, params, solve_ivp, t_eval):
    def model(t_eval, IC, params):
        Ea, R, k0, V, rho, Cp, deltaH, UA, q, cAf, Tf, Tc = params

        def k(T):
            return k0 * np.exp(-Ea / (R * T))

        def eq(t, y):

            cA, T = y
            dcAdt = (q / V) * (cAf - cA) - k(T) * cA
            dTdt = ((q / V) * (Tf - T)+ (-deltaH / rho / Cp) * k(T) * cA+ (UA / V / rho / Cp) * (Tc - T))

            return [dcAdt, dTdt]

        sol = solve_ivp(eq, [min(t_eval), max(t_eval)], IC, t_eval=t_eval)

        return sol

    solution = model(t_eval, IC, params)
    cA = solution.y[0]
    T = solution.y[1]
    return T, cA, solution


@app.cell
def _(T, cA, plt, solution):
    fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(10, 5))

    ax[0].plot(solution.t, cA, color="black", label="Concentration[mol/l")
    ax[0].set_ylabel("Concentration[mol/L]")
    ax[0].grid(True)
    ax[0].legend()

    ax[1].plot(solution.t, T, color="orange", label="Temperature[K]")
    ax[1].set_ylabel("Temperature[K]")
    ax[1].set_ylim(320, 370)

    plt.legend()
    plt.grid(True)
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


if __name__ == "__main__":
    app.run()
