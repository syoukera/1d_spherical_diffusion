import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time as tm
import cupy as cp

useGPU = False

D = 1.0
num_pnt = 101
length = 1.0
dr = length/(num_pnt-1)
allay = np.zeros(num_pnt)
radious = np.zeros(num_pnt)
# allay for gpu
allay_gpu = cp.zeros(num_pnt)
radious_gpu = cp.zeros(num_pnt)

# time for computation
dt =   0.000001
init = 0.0
maxt = 0.010
# time for visualize
vist = 0.0001
rest = 0.0

ims = []

def plot_allay(x, y):
    im = plt.plot(x, y, color="green")
    plt.xlim(0.0, 1.0)
    # plt.ylim(0.0, 1.0)
    ims.append(im)

# initialize
for i in range(num_pnt):
    radious[i] = length*(i)/(num_pnt-1)
allay[0:int(num_pnt*0.5)] = 1.0
# initalize gpu memory
radious_gpu = cp.asarray(radious)
allay_gpu = cp.asarray(allay)

#for idx, pnt in enumerate(allay):
#    print(idx, pnt)

# record start time of computation
start = tm.time()

# time-progress calculation
time = init
while(time < maxt):
    # calculate next-step
    if useGPU:
        allay_gpu[1:-1] = allay_gpu[1:-1] + dt*D*(((allay_gpu[2:] - 2 * allay_gpu[1:-1] + allay_gpu[:-2]) / dr ** 2) + 2/radious_gpu[1:-1] * ((allay_gpu[2:] - allay_gpu[:-2]) / dr))
    else:
        allay[1:-1] = allay[1:-1] + dt*D*(((allay[2:] - 2 * allay[1:-1] + allay[:-2]) / dr ** 2) + 2/radious[1:-1] * ((allay[2:] - allay[:-2]) / dr))
        

    #for idx, pnt in enumerate(allay):
    #    print(idx, pnt)

    # visualize
    rest = rest + dt
    if rest >= vist:
        if useGPU:
            allay = cp.asnumpy(allay_gpu)
        plot_allay(radious, allay)
        rest = 0.0

    time = time + dt

fig = plt.gcf()
ani = animation.ArtistAnimation(fig, ims, interval=100)
ani.save("figure/output.gif", writer="imagemagick")
# plt.show()

# record computational time
comt = tm.time() - start
print(f'Comutational time: {comt}')