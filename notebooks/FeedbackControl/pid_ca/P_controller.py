import marimo

__generated_with = "0.23.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.integrate import solve_ivp
    from dataclasses import dataclass, field

    return np, plt, solve_ivp


@app.cell
def _(np, solve_ivp):
    def process(t, y):
        K = 2
        tau = 20

        if t < 10:
            u = 0
        else:
            u = 5

        dydt = (K * u - y) / tau
        return dydt


    t = np.linspace(0, 200, 1000)

    sol = solve_ivp(
        process, [t[0], t[-1]], y0=[0], t_eval=t
    )  # function, t_span, y0, t_eval

    print(sol)
    return (sol,)


@app.cell
def _(np, plt, sol):
    u = np.zeros(200)
    u[10:] = 5

    fig, ax = plt.subplots()
    ax.plot(sol.t, sol.y[0], label="Step response")
    plt.legend(loc="center right")

    ax2 = ax.twinx()
    ax2.plot(u, color="black", linestyle="--", alpha=0.5, label="Step input")
    plt.legend()
    plt.show()
    return


@app.cell
def _():
    y_inf = 2 * 5  # K*delta(u)
    return


@app.cell
def _(np):
    K = 2
    tau = 20

    Kp = 10
    SP = 10

    dt = 0.1
    t_ = np.arange(0, 200, dt)

    PV = np.zeros(len(t_))
    MV = np.zeros(len(t_))
    ERROR = np.zeros(len(t_))
    return ERROR, K, Kp, MV, PV, SP, dt, t_, tau


@app.cell
def _(ERROR, K, Kp, MV, PV, SP, dt, t_, tau):
    # PROPORTIONAL CONTROLLER
    """The reason for a steady state error with P only is that as your system approaches the set-point the error signal gets smaller and smaller. Your control is Kp times that error signal and eventually the error will be small enough that Kp times the error won't be enough to force it all the way to zero."""

    for k in range(1, len(t_)):
        error = SP - PV[k - 1]
        MV[k] = (
            Kp * error
        )  # PID equation (MV0 = 0) -> Computing manipulated variable. The proportional controller sets the output                                                            # to P times the error
        ERROR[k] = error

        dPV = (
            K * MV[k] - PV[k - 1]
        ) / tau  # System equation -> computing change in process variable

        PV[k] = PV[k - 1] + dPV * dt  # Updating value of process variable

    # tau dPV + PV[k-1] = KKp(SP - PV[k-1])
    return


@app.cell
def _(K, Kp, SP):
    ess = SP / (1 + K * Kp)
    ess

    # Relative to the SP
    # ess/SP = 1/(1+K*Kp)
    return


@app.cell
def _(ERROR, MV, PV, SP, np, plt, t_):
    _fig, _ax = plt.subplots()
    _ax.plot(t_, PV, label="Process variable")
    _ax.plot(t_, np.ones_like(t_) * SP, label="Setpoint")
    plt.legend(loc="center right")

    _ax2 = _ax.twinx()
    _ax2.plot(
        t_,
        MV,
        color="black",
        linestyle="--",
        alpha=0.5,
        label="Manipulated variable",
    )
    _ax2.plot(t_, ERROR, color="red", linestyle="--", alpha=0.5, label="Error")
    plt.legend()
    plt.show()
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
