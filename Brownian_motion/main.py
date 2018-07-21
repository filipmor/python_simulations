import numpy as np
import matplotlib.pylab as plt
from mpl_toolkits.mplot3d import Axes3D

if __name__ == "__main__":

    # time in seconds
    t = 10
    # average number of motions in a time unit
    v = 10
    # number of steps in one trajectory
    n = v*t

    # Number of particles
    nop = 200

    # Initialize coordinates for all particles
    x = np.zeros((nop, n), dtype=np.float16)
    y = np.zeros((nop, n), dtype=np.float16)

    # Generate motion - random walk
    plt.figure(1)
    for i in range(nop):
        for j in range(1, n):
            x[i, j] = x[i, j-1] + np.random.normal(0)
            y[i, j] = y[i, j-1] + np.random.normal(0)
        plt.plot(x[i,:], y[i,:])
    plt.xlabel("X coord")
    plt.ylabel("Y coord")

    # Mean squared displacement
    msd = np.zeros((n,1))
    for i in range(n):
        for j in range(nop):
            msd[i,0] += x[j,i]**2 + y[j,i]**2
        msd[i,0] /= float(nop)

    plt.figure(2)
    plt.plot(msd)

    # Density distribution of particles
    R = np.zeros((nop, n))
    for i in range(nop):
        for j in range(n):
            R[i, j] = np.sqrt(x[i,j]**2 + y[i,j]**2)

    # Divide the space into rings - represent density distrubution
    # in the form of 3-D histogram
    dr = 1 # ring size
    binmax = 30 # max distance from the center of coordinate system
    binmin = 0 #  min distance from the center of coordinate system
    n_rings = np.linspace(binmin, binmax, num=dr*binmax)

    # calculate how density distribution changes in the following time steps
    count_fraction = np.zeros((n_rings.shape[0], n))
    for i in range(n):
        counts, bin_edges = np.histogram(R[:,i], n_rings)
        bin_n = 0
        for j in n_rings[0:-1]:
            if bin_n == 29:
                print("")
            count_fraction[bin_n, i] = counts[bin_n]/(np.pi*((j+1)**2 - j**2))
            bin_n += 1

    # Plot histogram 3D
    fig = plt.figure(3)
    ax = Axes3D(fig)
    ax.bar()
    #ax1.bar3d(n_rings, count_fraction)

    plt.show()

    pass