import marimo

__generated_with = "0.23.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import matplotlib.pyplot as plt

    return (plt,)


@app.function
# proportional control
def P(Kp, MV_bar=0):
    ''' Simulate a proportional controller.
    
    Arguments:
        Kp: the proportional gain
        MV_bar: the bias/offset

    Usage:
        proportional = P(Kp, MV_bar)
        next(proportional)
        MV = proportional.send((SP, PV))
    '''
    
    # initialize MV with bias/offset
    MV = MV_bar
    
    # run indefinitely
    while True:
        
        # limit the manipulated variable to the feasible range of values
        MV = max(0, min(100, MV))
        
        # yield MV to calling program. Then pause and wait for updates to SP and PV
        SP, PV = yield MV
        
        # compute error signal
        e = SP - PV
        
        # compute new value of the manipulated variable
        MV = MV_bar + Kp * e


@app.cell
def _(plt):
    from tclab import TCLab, clock, Historian, Plotter, setup

    TCLab = setup(connected=False, speedup=10)

    # create functions to simulate setpoints and disturbance variables
    def SP(t):
        return 40 if t >= 20 else 25

    def DV(t):
        return 100 if t >= 200 else 0

    # create a controller instance
    controller = P(100)

    # simulation duration and sampling time
    t_final = 600
    t_step = 2

    with TCLab() as lab:

        # intialize historian and plotting
        sources = [["T1", lambda: lab.T1],
                   ["SP1", lambda: SP(t)],
                   ["Q1", lab.Q1],
                   ["DV", lambda: DV(t)]]
        h = Historian(sources)

        # initialize maximum power for both heaters
        lab.P1 = 200
        lab.P2 = 200

        # initialize the controller
        U1 = next(controller)
        lab.Q1(U1)
        #lab.Q2(DV(0))

        # event loop
        for t in clock(t_final, t_step):

            # get measurement of process variable T1
            T1 = lab.T1

            # send current setpoint and PV to controller
            U1 = controller.send((SP(t), T1))

            # update manipulated variable
            lab.Q1(U1)

            # simulate disturbance
            lab.Q2(DV(t))

            # update the historian
            h.update(t)

    p = Plotter(h, t_final, layout=[["T1", "SP1"], ["Q1"], ["DV"]])
    p.update(t)
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
