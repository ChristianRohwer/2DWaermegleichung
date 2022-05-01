import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
%matplotlib inline
%config InlineBackend.figure_format='retina'

# Grösse der Metallplatte in mm
w = h = 10.
# Diskretisierung für x, y in mm
dx = dy = 0.1
# Thermische Leitfähigkeit von Stahl in mm2.s-1
D = 4.
# Temperaturen (Kelvin)
Tk, Th = 300, 700

nx, ny = int(w/dx), int(h/dy)

dx2, dy2 = dx*dx, dy*dy
dt = dx2 * dy2 / (2 * D * (dx2 + dy2))

u0 = Tk * np.ones((nx, ny))
u = u0.copy()

# Anfangsbedingungen: Ring mit innerem Radius r und Dicke, Mittelpunkt (cx,cy) (mm)
r, dr, cx, cy = 2, 1, 5, 5
ri2, ro2 = r**2, (r+dr)**2
for i in range(nx):
    for j in range(ny):
        p2 = (i*dx-cx)**2 + (j*dy-cy)**2
        if ri2 < p2 < ro2:
            u0[i,j] = Th

def do_timestep(u0, u):
    # Diskretisierung der Dynamik
    u[1:-1, 1:-1] = u0[1:-1, 1:-1] + D * dt * (
          (u0[2:, 1:-1] - 2*u0[1:-1, 1:-1] + u0[:-2, 1:-1])/dx2
          + (u0[1:-1, 2:] - 2*u0[1:-1, 1:-1] + u0[1:-1, :-2])/dy2 )

    u0 = u.copy()
    return u0, u

nsteps = 500 # Anzahl der Simulationsschritte
fig = plt.figure()
ax = fig.add_subplot()
# Anfangsbedingungen werden geplotted
im = ax.imshow(u0, cmap=plt.get_cmap('magma'), vmin=Tk, vmax=Th,
               interpolation='bicubic')
ax.set_axis_off()
ax.set_title('0.0 ms')
plt.show()

fig.subplots_adjust(right=0.85)
cbar_ax = fig.add_axes([0.9, 0.15, 0.03, 0.7])
cbar_ax.set_xlabel('$T$ / K', labelpad=20)
fig.colorbar(im, cax=cbar_ax)

plt.figure()
def animate(i):
    """Simulationschritt..."""
    global u0, u
    u0, u = do_timestep(u0, u)
    #ax.set_title('{:.1f} ms'.format(i*dt*1000))
    #im.set_data(u.copy())

interval = 10
ani = animation.FuncAnimation(plt.figure(), animate, frames=nsteps, repeat=True,
                              interval=interval,blit = True)
#f = r"YourDirectoryHere\animation.gif" 
#writergif = animation.PillowWriter(fps=30) 
#ani.save(f, writer=writergif)
plt.show()
