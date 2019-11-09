
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


D = 1.0
num_pnt = 101
length = 1.0
dr = length/(num_pnt-1)
allay = np.zeros(num_pnt)
radious = np.zeros(num_pnt)

dt = 0.00001
init = 0.0
maxt = 0.005

ims = []

# initialize
allay[0:int(num_pnt*0.5)] = 1.0
for i in range(num_pnt):
    radious[i] = length*(i)/(num_pnt-1)

#for idx, pnt in enumerate(allay):
#    print(idx, pnt)

# time-progress calculation
time = init
while(time < maxt):
    # calculate next-step
    allay[0] = 1.0
    allay[1:-1] = allay[1:-1] + dt*D*(((allay[2:] - 2 * allay[1:-1] + allay[:-2]) / dr ** 2) + 2/radious[1:-1] * ((allay[2:] - allay[:-2]) / dr))
    allay[-1] = 0.0

    #for idx, pnt in enumerate(allay):
    #    print(idx, pnt)

    # visualize
    fig = plt.figure()
    im = plt.plot(radious, allay)
    ims.append(im)

    time = time + dt

ani = animation.ArtistAnimation(fig, ims, interval=100)
plt.show()