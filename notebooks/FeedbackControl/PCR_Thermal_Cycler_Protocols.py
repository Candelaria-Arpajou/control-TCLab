import marimo

__generated_with = "0.23.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt

    n_cycles = 5
    activation = [(900,95)]
    cycling = [(20,94),(20,60),(30,72)]*n_cycles
    extension = [(600,72)]
    finish = [(0,30)]

    # concattenate into a list of time (time,temperature) intervals 
    protocol = np.concatenate([activation,cycling,extension,finish])
    protocol
    return np, plt, protocol


@app.cell
def _(np, plt, protocol):
    # Setpoint function 
    def PCR_setpoint(protocol,t,plot = True):
    
        ramp_rate = 0.5 # deg/sec
        time_now = 0.0
        temp_now = 21.0
    
        SP_list = [[time_now, temp_now]]
    
        for time, temp in protocol:
            time_now += np.abs((temp - temp_now)/ramp_rate)
            temp_now = temp
            SP_list.append([time_now,temp_now])
    
            time_now += time
            SP_list.append([time_now, temp_now])
    
        SP_array = np.array(SP_list)
    
        def ST(t):
            return np.interp(t,SP_array[:,0], SP_array[:,1])

        if plot == True:
            fig, _ax = plt.subplots(1, 1, figsize=(12, 5))
            _ax.plot(t,ST(t))
            _ax.set_title("Setpoint function")
            plt.show()

    t_new = np.linspace(0,3000,3000)
    PCR_setpoint(protocol,t_new)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
