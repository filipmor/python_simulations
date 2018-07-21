from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pylab as plt
from matplotlib import cm


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
    z = np.zeros((nop, n), dtype=np.float16)

    # Generate motion - random walk
    fig = plt.figure(1)
    ax = fig.gca(projection='3d')
    for i in range(nop):
        for j in range(1, n):
            x[i, j] = x[i, j-1] + np.random.normal(0)
            y[i, j] = y[i, j-1] + np.random.normal(0)
            z[i, j] = z[i, j-1] + np.random.normal(0)
        ax.plot(x[i,:], y[i,:], z[i,:])
    ax.set_xlabel("X coord")
    ax.set_ylabel("Y coord")
    ax.set_zlabel("Z coord")


    # Mean squared displacement
    msd = np.zeros(n)
    for i in range(n):
        for j in range(nop):
            msd[i] += x[j,i]**2 + y[j,i]**2 + z[j,i]**2
        msd[i] /= float(nop)

    plt.figure(2)
    plt.plot(msd)
    plt.xlabel("time [s]")
    plt.ylabel("MSD")

    # Density distribution of particles
    R = np.zeros((nop, n))
    for i in range(nop):
        for j in range(n):
            R[i, j] = np.sqrt(x[i,j]**2 + y[i,j]**2 + z[i,j]**2)

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
    fig = plt.figure()
    ax1 = fig.add_subplot(111, projection='3d')
    XX, YY = np.meshgrid(np.arange(0, n, 1), n_rings)
    print(XX.shape)
    print(YY.shape)
    print(count_fraction.shape)
    ax1.plot_surface(XX, YY, count_fraction, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    ax1.set_xlabel("time (s)")
    ax1.set_ylabel("distance from the center")
    ax1.set_zlabel("counts")

    # Autocorrelation with normalization
    x_traj = np.cumsum(x[50,:])/n
    y_traj = np.cumsum(y[50,:])/n
    z_traj = np.cumsum(z[50,:])/n
    traj = np.zeros(n)
    for i in range(n):
        traj[i] = np.sqrt(x_traj[i]**2 + y_traj[i]**2 + z_traj[i]**2)
    autocorr = np.correlate(traj, traj, mode="full")
    fig = plt.figure()
    plt.plot(np.arange(0,n,1), autocorr)

    plt.show()
