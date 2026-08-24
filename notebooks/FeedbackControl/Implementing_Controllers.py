import marimo

__generated_with = "0.23.0"
app = marimo.App(width="medium")


@app.function
def relay_with_deadzone(PV, SP, MV_prev, MV_min, MV_max, d):
    if PV <= SP - d:
        MV = MV_max
    elif PV >= SP + d:
        MV = MV_min
    else:
        MV = MV_prev
    return MV


@app.function
def Relay(MV_min=0, MV_max=100, d=0):
    MV = MV_min
    while True:
        SP, PV = yield MV
        if PV <= SP - d:
            MV = MV_max
        if PV >= SP + d:
            MV = MV_min


@app.cell
def _():
    from tclab import TCLab, clock, Historian, Plotter, setup
    import matplotlib.pyplot as plt

    TCLab = setup(connected=False, speedup=20)

    # control parameters
    controller = Relay(MV_min=0, MV_max=100, d=0.5)
    next(controller)

    U_min = 0
    U_max = 100
    T_SP = 40
    d = 0.5

    t_final = 250

    with TCLab() as lab:
        lab.P1 = 200
        h = Historian(lab.sources)
        #U1 = U_min
        for t in clock(t_final):
            T1 = lab.T1
            #U1 = relay_with_deadzone(T1, T_SP, U1, U_min, U_max, d)
            U1 = controller.send((T_SP, T1))
            lab.Q1(U1)
            h.update(t)

    p = Plotter(h, t_final)
    p.update(t_final)
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
