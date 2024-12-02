from mpl_toolkits.axes_grid1 import ImageGrid
import matplotlib.pyplot as plt
 
fig = plt.figure(figsize=(40, 35))
grid2 = ImageGrid(
        fig, 111, nrows_ncols=(4, 3), axes_pad=2
    )
# plt.tight_layout()
plt.savefig("./gridtest.png")

# plt.subplot(111)  # 1 row, 1 column, 1st plot
# plt.plot([1, 2, 3], [4, 5, 6])
# plt.savefig("./gridtest2.png")

