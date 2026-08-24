import marimo

__generated_with = "0.23.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _():
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.integrate import solve_ivp
    from dataclasses import dataclass, field

    return dataclass, field, np, plt, solve_ivp


@app.cell
def _(np, solve_ivp, t):
    def process(_t, y):
        K = 2
        tau = 20

        if _t < 10:
            u = 0
        else:
            u = 5

        dydt = (K * u - y) / tau
        return dydt


    _t = np.linspace(0, 200, 1000)

    sol = solve_ivp(
        process, [t[0], _t[-1]], y0=[0], t_eval=_t
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
    Kp = 2
    Ki = 1/5
    Kd = 0.05
    dt = 0.1

    t = np.arange(0, 400, dt)
    SP = np.where(t < 100, 2, np.where(t < 200,4,1))

    PV = np.zeros(len(t))
    MV = np.zeros(len(t))
    ERROR = np.zeros(len(t))
    return K, Kd, Ki, Kp, MV, PV, SP, dt, t, tau


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### PI controller
    """)
    return


@app.cell
def _(K, Kd, Ki, Kp, MV, PV, SP, dataclass, dt, field, np, t, tau):
    @dataclass
    class PIController:
        Kp: float
        Ki: float
        Kd: float
        dt: float

        MV_min: float
        MV_max: float

        MV: float = field(init=False, default=0.0)
        ek_1: float = field(init=False, default=0.0)
        ek_2: float = field(init=False, default=0.0)

        def update(self, SP, PV):
            e = SP - PV
            self.MV += (
                self.Kp * (e - self.ek_1) + 
                self.Ki * self.dt * e + 
                (self.Kd / self.dt) * (e - 2 * self.ek_1 + self.ek_2)
            )
        
            self.MV = np.clip(self.MV, self.MV_min, self.MV_max)
            self.ek_1 = e
            self.ek_2 = self.ek_1
        
            return self.MV

    controller = PIController(Kp=Kp, Ki=Ki, Kd=Kd, dt=dt, MV_min=0, MV_max=100)

    for k in range(1, len(t)):
        MV[k] = controller.update(SP[k], PV[k-1])
        dPV = (K * MV[k] - PV[k-1]) / tau
        PV[k] = PV[k-1] + dt * dPV
    return


@app.cell
def _(MV, PV, SP, np, plt, t):
    _fig, _ax = plt.subplots()
    _ax.plot(t, PV, label="Process variable")
    _ax.plot(t, np.ones_like(t) * SP, label="Setpoint")
    plt.legend(loc="center right")

    _ax2 = _ax.twinx()
    _ax2.plot(
        t,
        MV,
        color="black",
        linestyle="--",
        alpha=0.5,
        label="Manipulated variable",
    )

    plt.legend()
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
