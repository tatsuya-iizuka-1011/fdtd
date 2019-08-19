import numpy as np
import matplotlib.pyplot as plt 
import matplotlib.animation as animation

from tqdm import tqdm 

permittivity_of_vacuum = 8.85419e-12
permeability_of_vacuum = 1.25664e-06

# set 
t = 0
dt = 1e-8
nt= 1e5

ndim = 2
Nx = Ny = 100
dx = dy = 20 # when wave length is 1.5e6

# set field properties
conductivity = 0 * np.ones((Nx, Ny, 1))
permittivity = permittivity_of_vacuum * np.ones((Nx, Ny, 1))
permeability = permeability_of_vacuum * np.ones((Nx, Ny, 1))

coeff1 = conductivity * dt / (2 * permittivity)
c_ez = (1 - coeff1) / (1 + coeff1) 
c_ezlx = dt/(permittivity * dx * (1 + coeff1))
c_ezly = dt/(permittivity * dy * (1 + coeff1))

c_hxlx = dt/(permeability * dx)
c_hxly = dt/(permeability * dy)

e_field = np.zeros((Nx, Ny, 1)) # only z direction
h_field = np.zeros((Nx, Ny, 2)) # x and y direction

# set excitation properties
ex_x = int(Nx / 2)
ex_y = int(Ny / 2)
t_arr  = np.arange(nt) * dt
f = 2 * 1e5
ez_val = 10 * np.sin(2 * np.pi * f * t_arr)
ez_result = np.zeros((t_arr.shape[0], Nx, Ny, 1))

for k, ex_ez in tqdm(enumerate(ez_val)):
    # excitation
    e_field[ex_x, ex_y] = ex_ez
    ez_result[k] = e_field
    # calc Ez
    for i  in range(1,Nx):
        for j  in range(1,Ny):5630    
        e_field[i, j] = c_ez[i, j] * e_field[i,j] + c_ezly[i,j]*(h_field[i,j,0] - h_field[i,j-1,0])  + c_ezlx[i,j]*(h_field[i,j,1] - h_field[i-1,j,1])
    # print(e_field[ex_x-1:ex_x+2,ex_y-1:ex_y+2])
    # calc Hx
    for i  in range(0, Nx):
        for j  in range(0, Ny - 1):
            h_field[i,  j, 0] = h_field[i,  j, 0] - c_hxly[i, j] * (e_field[i, j+1] - e_field[i, j])
    # calc Hy
    for i  in range(0, Nx - 1):
        for j  in range(0, Ny):
            h_field[i,  j, 1] = h_field[i,  j, 1] + c_hxlx[i, j] * (e_field[i+1, j] - e_field[i, j])
    #plt.plot(t_arr, ez_val)


fig = plt.figure()
ims = []
res_len = ez_val.shape[0]

for k in range(res_len):
    im = plt.imshow(ez_result[k,:,:,0], animated=True)
    ims.append([im])

ani = animation.ArtistAnimation(fig, ims, interval=50, blit=True, repeat_delay=1000)
ani.save('anim.gif', writer="imagemagick")