import marimo

__generated_with = "0.23.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _():
    import tclab
    import matplotlib.pyplot as plt
    import numpy as np

    return np, plt, tclab


@app.cell
def _():
    # https://tclab.readthedocs.io/en/latest/index.html
    # https://ndcbe.github.io/controls/notebooks/2/First-Order-Model-for-a-Single-Heater.html
    # https://github.com/ndcbe/controls
    # https://abe-mart.github.io/tclab-sim/
    return


@app.cell
def _(tclab):
    TCLab = tclab.setup(connected=False, speedup=100)

    stime = 600  # Simulation time

    with TCLab() as lab:
        h = tclab.Historian(lab.sources)
        for t in tclab.clock(stime):
            lab.Q1(0 if t <= 50 else 50)
            # print("Time:", t, 'seconds')
            h.update(t)

    t, T1, T2, Q1, Q2 = h.fields
    return Q1, T1, t


@app.cell
def _(Q1, T1, plt, t):
    fig, ax1 = plt.subplots()
    ax1.plot(t, T1, label="T1", color="orange")
    # ax1.plot(t, T2, label="T2")
    # ax1.plot(t, T1_ + Tamb)
    # ax1.plot(t, T2)
    ax1.set_ylabel("Temperature / °C")
    ax1.legend(loc="right")

    ax2 = ax1.twinx()
    ax2.plot(t, Q1, label="Q1")
    # ax2.plot(t, Q2, label="Q2")
    ax2.set_ylabel("Input signal / %")
    ax2.set_xlabel("Time / seconds")
    ax2.legend()

    plt.grid()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Estimating alpha
    """)
    return


@app.cell
def _():
    P1 = 200
    u1 = 50

    voltage = 5.1  # volts
    intensity = 0.315  # amps

    power = voltage * intensity

    # power = alpha x P1 x U1
    alpha = power / (P1 * u1)  # watts/(unitsP1 x unitsU1)
    alpha
    return P1, alpha, u1


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Estimating Ua
    """)
    return


@app.cell
def _(P1, T1, alpha, u1):
    Tamb = T1[0]
    Ua = alpha * P1 * u1 / (T1[-1] - Tamb)
    Ua
    return Tamb, Ua


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Estimating max power & max temp.
    """)
    return


@app.cell
def _(alpha):
    maxP1 = 255
    maxu1 = 100
    maxPower = alpha * (maxP1 * maxu1)
    maxPower
    return (maxPower,)


@app.cell
def _(Tamb, Ua, maxPower):
    maxT = Tamb + maxPower / Ua
    maxT
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Apply to Model Equation
    """)
    return


@app.cell
def _(P1, Q1, T1, Tamb, Ua, alpha, np, t):
    Cp = 6
    a = -Ua / Cp
    b = alpha * P1 / Cp

    T1_0 = T1[0] - Tamb
    T1_ = alpha * P1 / Ua

    exp_arg = np.roll(np.array(t) - t[0], 50)
    exp_arg[:50] = 0

    T1_t = T1_ * np.array(Q1) + (T1_0 - T1_*np.array(Q1)) * np.exp(a * exp_arg)
    return (T1_t,)


@app.cell
def _(Q1, T1, T1_t, Tamb, plt, t):
    fig_, ax_ = plt.subplots()
    ax_.plot(t, T1)
    ax_.plot(t, T1_t + Tamb)
    #ax_.plot(t, T2)
    ax_.set_ylabel("Temperature / °C")

    ax_2 = ax_.twinx()
    ax_2.plot(t, Q1)
    # ax_2.plot(t, Q2)

    ax_2.set_xlabel("Time / seconds")

    plt.grid()
    plt.show()
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
