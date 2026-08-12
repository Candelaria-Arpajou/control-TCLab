import marimo

__generated_with = "0.23.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import matplotlib.pyplot as plt

    return (plt,)


@app.cell
def _():
    # A generator is an elegant way for creating an iterator
    # yield pausses the excution of the function and returns de value, it continues when it sees the next value of the iterator. The yield key basically turns the fonction into a generator? The final purpose is to be able to loop through a fonction without having to store all the data
    return


@app.cell
def _(plt):
    from tclab import TCLab, clock, Historian, Plotter, setup

    TCLab = setup(connected=False, speedup=10)

    def SP(t):
        return 40 if t >= 20 else 25

    def DV(t):
        return 100 if t >= 200 else 0

    def P(Kp, MV_bar=0):
        MV = MV_bar

        while True:
            MV = max(0,min(100,MV))
            SP, PV = yield MV
            e = SP - PV
            MV = MV_bar + Kp * e

    def PI(Kp, Ki, MV_bar=0):
        MV = MV_bar
        e_prev = 0

        while True:
            SP, PV = yield MV
            e = SP - PV
            MV = MV + Kp*(e - e_prev) + t_step*Ki*e
            MV = max(0,min(100,MV))

            e_prev = e

    #controller = P(100)
    controller = PI(3, 0.2)

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
        lab.Q2(DV(0))

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


if __name__ == "__main__":
    app.run()
