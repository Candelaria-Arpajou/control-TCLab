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
    import pandas as pd

    import matplotlib.pyplot as plt

    # specify a setpoint profile
    profile = [
        (0, 25),
        (50, 35),
        (120, 35),
        (120, 40),
        (170, 50),
        (170, 50),
        (270, 50),
        (370, 40),
        (450, 40),
        (600, 25),
    ]

    sp_profile = pd.DataFrame(profile, columns=["Time", "SP"])
    ax = sp_profile.plot(x="Time", grid=True, ylim=(25, 60), title="Setpoint")
    ax.annotate("Step", xy=(120, 37.5), xytext=(180, 37.5), fontsize=12, 
                va="center", arrowprops=dict(facecolor='black', shrink=0.05))
    ax.annotate("Ramp", xy=(320, 45), xytext=(380, 48), fontsize=12, 
                va="center", arrowprops=dict(facecolor='black', shrink=0.05))
    ax.annotate("Soak/Dwell", xy=(210, 50), xytext=(210, 55), fontsize=12, 
                ha="center", arrowprops=dict(facecolor='black', shrink=0.05))
    return np, pd, plt, sp_profile


@app.cell
def _(pd, sp_profile):
    sp_profiles = pd.DataFrame([(0,25),(50,35),(120,35),(120,40),(170,50),(170,50),(270,50),(370,40),(450,40),
    (600,25)],columns=["Time","SP"])

    print(sp_profile)
    sp_profile.plot(x="Time", y='SP', style={"SP":"r"}, ms=10,ylim=(25,80),grid=True)
    return


@app.cell
def _(np, sp_profile):
    def sp(t):
        t_interp = sp_profile["Time"]
        y_interp = sp_profile["SP"]
        return np.interp(t,t_interp,y_interp)

    sp(500)
    return (sp,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Example of setpoint function
    """)
    return


@app.cell
def _(sp_profile):
    sp_profile
    return


@app.cell
def _(np, sp_profile):
    def create_setpoint_function(profile):
        profile = np.array(profile)
        t_interp = profile[:,0]
        y_interp = profile[:,1]

        def setpoint_function(t):
            return np.interp(t,t_interp, y_interp) # t: The x-coordinates at which to evaluate the interpolated values.
                                                   # t_interp: The x-coordinates of the data points
                                                   # y_interp: The y-coordinates of the data points

        return setpoint_function

    sp1 = create_setpoint_function(sp_profile)
    sp1(200)
    return create_setpoint_function, sp1


@app.cell
def _(sp):
    t = 100
    print(f"At time = {t:3d}, setpoint = {sp(t)}")
    return


@app.cell
def _(create_setpoint_function, np, plt, sp1, sp_profile):
    # compute setpoint values
    t_ = np.linspace(0,600,600)

    sp_profile.loc[5:6,"SP"] = [95, 95]
    sp2 = create_setpoint_function(sp_profile)
    y = sp2(t_)

    fix, ax_ = plt.subplots(1,1,figsize=(10,5))
    ax_.plot(t_,sp1(t_))
    ax_.plot(t_,sp2(t_))
    ax_.set_label("Time / seconds")
    ax_.set_title("setpoint function")
    ax_.grid(True)
    plt.show()

    return


@app.cell
def _(create_setpoint_function, np, plt):
    T_amb = 21.0

    sp_1 = create_setpoint_function([[0, T_amb], [20, T_amb], [60, 50], [100, 50], [140, T_amb]])
    sp_2 = create_setpoint_function([[0, T_amb], [0, 45], [120, 35], [200, T_amb]])

    # create plot axes
    fig, _ax = plt.subplots(2, 1)

    # plot setpoint functions
    _t = np.linspace(-1, 250, 250)
    _ax[0].plot(_t, sp_1(_t))
    _ax[1].plot(_t, sp_2(_t))
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
