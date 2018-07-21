from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import time

if __name__ == "__main__":
    # Geometrical and physical parameters of a sheet
    A = 0.2 # length of sheet
    B = 0.02 # length of heater
    h = 0.005 # thickness of a sheet
    cwS = 450.0 # specific heat for steel
    roS = 7860.0 # density of the steel
    KS = 58.0 # heat transmittance for the steel
    cwAl = 900.0# specific heat for aluminium
    roAl = 2700.0 # density of aluminium
    KAl = 237.0 # heat transmittance for the aluminium
    T0 = 293.0 # initial temperature of the environment
    Tr1 = 283.0 # boundary temp r1
    Tr2 = 353.0 # boundary temp r2

    t_max = 10.0 # time of simulations

    # Parameters for heat equation algorithm
    NX = 101 # number of nodes on X axis
    NY = 101 # number of nodes on Y axis
    NT = 1001 # number of time steps

    dx = A/NX
    dy = A/NY
    dt = t_max/NT

    # span of the heater
    xl = int(np.floor((A-B)/(2*dx)))
    xr = NX-xl
    yl = int(np.floor((A-B)/(2*dy)))
    yr = NY-yl

    # Output table of temperatures
    T = np.zeros((NX, NY), dtype=np.float16)
    T_aux = np.array(T) # auxiliary table for updates

    # Initial condition
    T[:,:] = T0

    # Choosing the material to analyse
    cw = cwAl
    ro = roAl
    K = KAl

    # Boundary conditions
    T_aux[xl:xr, yl:yr] = 1.0 # excluding from calculations edges of heater
    T[0, :] = Tr1
    T[-1, :] = Tr1
    T[:, 0] = Tr1
    T[:, -1] = Tr1
    T[xl:xr, yl:yr] = Tr2

    # Constants of the derivative equation
    Cx = K*dt/(cw*ro*dx**2)
    Cy = K*dt/(cw*ro*dy**2)

    # Calculations
    XX, YY = np.meshgrid(np.arange(0, A, dx), np.arange(0,A,dy))

    T1 = np.array(T) # additional auxiliary matrix

    plt.ion()
    fig = plt.figure()
    ax = fig.gca(projection="3d")

    for n in range(NT):
        #plt.clf()
        T = np.array(T1)
        for i in range(1, NX-1):
            for j in range(1, NY-1):
                if(T_aux[i,j]==0.0):
                    T1[i,j] = T[i,j] + Cx * (T[i-1,j] - 2.0*T[i,j] + T[i+1,j]) + Cy * (T[i,j-1] - 2.0*T[i,j] + T[i,j+1])
            #print(T1[i,j])
        Z = T1.reshape(XX.shape)
        surf = ax.plot_surface(XX, YY, Z, cmap=cm.coolwarm, linewidth=0, antialiased=False)
        fig.canvas.draw()
        #print("Draw")
        #plt.show()
        time.sleep(0.1)

    #Z = T1.reshape(XX.shape)
    #surf = ax.plot_surface(XX, YY, Z, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    #plt.show()
