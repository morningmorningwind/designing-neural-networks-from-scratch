# encoding: utf-8

import pylab as plt
import numpy as np
from matplotlib import animation
from IPython.display import HTML
import tempfile

def animate(datalist, file_name=None, interval=100, figsize=(3,3), frames=None, dpi=50, **kwargs):
    fig, ax = plt.subplots(figsize=figsize)
    if file_name is None:
        file_name = tempfile.NamedTemporaryFile().name+'.gif'
    else:
        if file_name[-4:] != '.gif': file_name += '.gif'

    im = ax.imshow(datalist[0],cmap=plt.get_cmap('rainbow'), **kwargs)
    
    def init():
        im.set_data(datalist[0])
        return im,

    def animate_gen(i):
        im.set_data(datalist[i])
        return im,
    if frames is None:
        frames = len(datalist)
    anim = animation.FuncAnimation(fig, animate_gen, frames=frames, init_func = init, interval=interval, blit=True, repeat=True, repeat_delay=500)
    anim.save(file_name,writer='imagemagick',dpi=dpi)
    plt.close(fig)
    del anim
    return HTML('<img src="'+file_name+'"></img>')

