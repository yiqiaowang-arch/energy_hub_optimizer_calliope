from textwrap import fill
import matplotlib.pyplot as plt
from matplotlib.sankey import Sankey

# Plot settings
plt.rcParams["font.family"] = "Roboto"
plt.rcParams["xtick.direction"] = "in"
plt.rcParams["ytick.direction"] = "in"
plt.rcParams["legend.loc"] = "lower center"
plt.rcParams["legend.frameon"] = False
plt.rcParams["font.size"] = 12
plt.rcParams["axes.titlesize"] = 16
plt.rcParams["axes.titley"] = 1.03
plt.rcParams["axes.linewidth"] = 0
plt.rcParams["xtick.major.width"] = 2  # setting the x-axis tick width globally
plt.rcParams["ytick.major.width"] = 2  # setting the y-axis tick width globally
# Set fig size
plt.rcParams["figure.figsize"] = (12, 8)

# flows = [2958, -119, -441, -616, -131, -1651]
flows = [2839, -131, -441, -616, -1651]
# labels = [
#     "Total",
#     "outside bound",
#     "non-res",
#     "invalid",
#     "within networks",
#     "analysed",
# ]
labels = [
    "Total",
    "within networks",
    "invalid",
    "non-res",
    "analysed",
]
orientations = [0, -1, -1, -1, 0]
Sankey(
    flows=flows,
    labels=labels,
    orientations=orientations,
    scale=1 / 3000,
    # trunklength=2.5,
    # offset=0.5,
    pathlengths=[0.3, 0.5, 0.5, 0.5, 0.3],
    fill=False,
).finish()
# fig, ax = plt.subplots()
# sankey = Sankey(scale=1 / 10000)
# sankey.add(
#     flows=flows,
#     labels=labels,
#     orientations=orientations,
#     # trunklength=1,
# )
# plt.title("Sankey diagram of buildings to be analysed")
plt.tight_layout()
plt.show()
