import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("data/tclab-data.csv", sep = ',')

print(data)

data.plot(y=["T1", "T2"], 
          #title=f"{P1=}, {U1=}",
          xlabel="seconds", 
          ylabel="deg C", 
          grid=True, 
          figsize=(8, 2.5)
         )

data.plot(y=["Q1", "Q2"],
          #title=f"{P1=}, {U1=}",
          xlabel="seconds", 
          ylabel="% of full range", 
          grid=True, 
          figsize=(8, 1.5),
          ylim=(0, 100)
         )

plt.show()