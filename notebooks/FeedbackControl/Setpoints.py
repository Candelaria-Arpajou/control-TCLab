import marimo

__generated_with = "0.23.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return


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
    return np, pd, sp_profile


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
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
