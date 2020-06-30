import matplotlib.pyplot as plt
from matplotlib import animation
import numpy as np


fig = plt.figure()
fig.set_dpi(200)
fig.set_size_inches(2, 2)

ax = plt.axes(xlim=(-10, 120), ylim=(-10, 120))
rect = plt.Rectangle((10, 10), 100, 100, fill=False, fc='b')
ax.add_patch(rect)
ax.text(-6, 4, '00')
ax.text(114, 4, '10')
ax.text(-6, 114, '01')
ax.text(114, 114, '11')


bit_val = {(10, 10):0, (10, 110):1,(110, 10):2,(110, 110):3}
pos_val = dict((node,pos) for pos,node in bit_val.items())
circle = plt.Circle((10, 10), radius=3, fc='r')

ax.axis('equal')
ax.axis('off')

def init():
    circle.center = (10,10)
    ax.add_patch(circle)
    return circle,

def animate(i):
    x, y = circle.center
    old_node = bit_val[x,y]
    coin_flip =np.random.randint(2)
    if coin_flip:
        new_node = old_node^1
    else:
        new_node = old_node^2

    circle.center = pos_val[new_node]
    return circle,

anim = animation.FuncAnimation(fig, animate,
                               init_func=init,
                               frames=360,
                               interval=30,
                               blit=True)
anim.save('square.gif', fps=1)
plt.show()

