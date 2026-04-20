import marimo

__generated_with = "0.23.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.integrate import solve_ivp
    from scipy.optimize import least_squares

    return least_squares, np, pd, plt, solve_ivp


@app.cell
def _(pd):
    data_file = "data/Step_Test_Data.csv"
    data = pd.read_csv(data_file, sep=",")
    data = data.set_index("Time")
    data = data.drop(["Unnamed: 0.2", "Unnamed: 0", "Unnamed: 0.1"], axis=1)
    data.head()
    return (data,)


@app.cell
def _(data, np):
    Tamb = 21  # deg C
    alpha = 0.00016  # watts / (units P1 * percent U1)
    P1 = 200  # P1 units

    # input values
    U1 = 50  # steady state value of u1 (percent)

    # extract data from experiment
    t_expt = data.index

    initial_guess = [5, 1, 0.05, 0.05]
    TS1_mes = np.asarray(data["TS1_measured"]).flatten() # Time is the index of the DataFrame, so TS1_mes in indexed
    return P1, TS1_mes, Tamb, U1, alpha, initial_guess, t_expt


@app.cell
def _(
    P1,
    TS1_mes,
    Tamb,
    U1,
    alpha,
    initial_guess,
    least_squares,
    np,
    solve_ivp,
    t_expt,
):
    def two_ssm(A,B,C):
        IC_ = [0, 0] # ---> TH1 = Tamb & TS1 = Tamb, then TH1 - Tamn = 0 & TS1 - Tamb = 0

        def u(t):
            return np.array([U1])

        def eq(t, x):
            dx = A @ x + B @ u(t)
            # y = C @ x
            return dx

        results = solve_ivp(eq, [min(t_expt), max(t_expt)], IC_, t_eval=t_expt)
        sol = results.y[1] + Tamb
        return sol # ----> predictions for TS1

    def pred_error(params,y_mes):
        CpH, CpS, Ub, Ua = params

        A = np.array([[-(Ua + Ub) / CpH, Ub / CpH], [Ub / CpS, -Ub / CpS]])
        B = np.array([[alpha * P1 / CpH, 0]]).T
        C = np.array([[0, 1]])

        y_pred = two_ssm(A,B,C)
        return y_pred - y_mes

    results = least_squares(pred_error, initial_guess , args = (TS1_mes,), loss="arctan")
    return results, two_ssm


@app.cell
def _(results):
    CpH, CpS, Ua, Ub = results.x
    print(f"CpH = {CpH},  CpS = {CpS},   Ua = {Ua},  Ub = {Ub}")
    return CpH, CpS, Ua, Ub


@app.cell
def _(CpH, CpS, P1, Ua, Ub, alpha, data, np, plt, two_ssm):
    new_params = CpH, CpS, Ua, Ub

    A = np.array([[-(Ua + Ub)/CpH, Ub/CpH], [Ub/CpS, -Ub/CpS]])
    B = np.array([[alpha * P1 / CpH, 0]]).T
    C = np.array([[0, 1]])

    data["TS1_pred"] = two_ssm(A,B,C)

    plt.plot(data["TS1_measured"])
    plt.plot(data["TS1_pred"])

    plt.legend(["TS1 (measured)", "TS1 (predicted)"])
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
