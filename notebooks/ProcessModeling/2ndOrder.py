import marimo

__generated_with = "0.23.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt

    return mo, np, pd, plt


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Higher-order state space model to the step test data (two-state model)
    """)
    return


@app.cell
def _(pd):
    data_file = "data/Step_Test_Data.csv"
    data = pd.read_csv(data_file, sep=",")
    data = data.set_index("Time")
    data = data.drop(["Unnamed: 0.2", "Unnamed: 0", "Unnamed: 0.1"], axis=1)
    data.head()
    return (data,)


@app.cell
def _(data, plt):
    data.plot(
        y=["TS1_measured", "TS2_measured"],
        figsize=(10, 3),
        grid=True,
        ylabel="deg C",
        xlabel="Time (s)",
        linestyle="",
        marker=".",
    )
    plt.legend(["TS1 (measured)", "TS2 (measured)"])

    data.plot(
        y=["Q1"],
        figsize=(10, 3),
        grid=True,
        ylabel="% of power range",
        ylim=(-5, 105),
        xlabel="Time (s)",
    )
    plt.show()
    return


@app.cell
def _(data):
    # known parameters
    Tamb = 21  # deg C
    alpha = 0.00016  # watts / (units P1 * percent U1)
    P1 = 200  # P1 units

    # adjustable parameters
    CpH = 5  # joules/deg C
    CpS = 1  # joules/deg C
    Ua = 0.05  # watts/deg C
    Ub = 0.05  # watts/deg C

    # input values
    U1 = 50  # steady state value of u1 (percent)

    # extract data from experiment
    t_expt = data.index
    return CpH, CpS, P1, Tamb, U1, Ua, Ub, alpha, t_expt


@app.cell
def _(t_expt):
    t_expt
    return


@app.cell
def _(CpH, CpS, P1, Tamb, U1, Ua, Ub, alpha, data, t_expt):
    from scipy.integrate import solve_ivp

    def ode2(CpH, CpS, Ua,Ub): 
        # initial conditions
        TH1 = Tamb
        TS1 = Tamb
        IC = [TH1, TS1]

        def twos_ode(t,y):
            TH1, TS1 = y
            dTH1dt = Ua/CpH*(Tamb - TH1) + Ub/CpH*(TS1 - TH1) + alpha*P1*U1/CpH
            dTS1dt = Ub/CpS*(TH1 - TS1)

            return [dTH1dt, dTS1dt]

        results = solve_ivp(twos_ode,[min(t_expt), max(t_expt)],IC, t_eval = t_expt)

        return results

    sol = ode2(CpH, CpS, Ua,Ub)

    data["TH1_pred"] = sol.y[0]
    data["TS1_pred"] = sol.y[1]
    return (solve_ivp,)


@app.cell
def _(data, plt):
    plt.plot(data["TS1_measured"])
    plt.plot(data["TH1_pred"])
    plt.plot(data["TS1_pred"])

    plt.legend(["TS1 (measured)", "TH1 (predicted)", "TS1 (predicted)"])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Vector notation
    """)
    return


@app.cell
def _(CpH, CpS, P1, Ua, Ub, alpha, np):
    A = np.array([[-(Ua + Ub)/CpH, Ub/CpH], [Ub/CpS, -Ub/CpS]])
    B = np.array([[alpha * P1/CpH, 0]]).T
    C = np.array([[0, 1]])

    B_ = np.array([[alpha*P1/CpH], [0]])
    return A, B, C


@app.cell
def _(A, B, C, U1, np, solve_ivp, t_expt):
    # IC = initial condition of the state variables TH1 = Tamb and TS1 = Tamb, then IC = [TH& - Tamb,TS1 - Tamb] = [0,0]
    def u(t):
        return np.array([U1])

    def two_ssm(A, B, C):
        IC_ = [0, 0]

        def eq(t, x):
            dx = A @ x + B @ u(t)
            #y = C @ x
            return dx

        results = solve_ivp(eq, [min(t_expt), max(t_expt)], IC_, t_eval=t_expt)
        return results


    sol2 = two_ssm(A, B, C)
    sol2
    return (sol2,)


@app.cell
def _(Tamb, sol2):
    x1 = sol2.y[0]
    x2 = sol2.y[1]

    TH1_pred = x1 + Tamb
    TS1_pred = x2 + Tamb
    return TH1_pred, TS1_pred


@app.cell
def _(TH1_pred, TS1_pred, data, plt):
    plt.plot(data["TS1_measured"])
    plt.plot(TH1_pred)
    plt.plot(TS1_pred)

    plt.legend(["TS1 (measured)", "TH1 (predicted)", "TS1 (predicted)"])
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
