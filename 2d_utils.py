import numpy as np
from tqdm import tqdm
from IPython.display import clear_output

def through_pixels(p0, p1):
    d = max(int(((p0[0]-p1[0])**2 + (p0[1]-p1[1])**2) ** 0.5), 1)

    pixels = p0 + (p1-p0) * np.array([np.arange(d+1), np.arange(d+1)]).T / d
    pixels = np.unique(np.round(pixels), axis=0).astype(int)

    return pixels

def build_through_pixels_dict(hooks):
    """
    Build a dictionary of through pixels for each pair of hooks.
    """
    for i in range(hooks.shape[0]):
        for j in range(i+1, hooks.shape[0]):
            p0, p1 = hooks[i], hooks[j]
            d[(i,j)] = through_pixels(p0, p1)
    return d

def build_through_pixels_dict_old(hooks):
    n_hooks = hooks.shape[0]
    n_hook_sides = n_hooks * 2

    print(f"n_hooks = {n_hooks}")
    print(hooks.shape)
    l = [(0,1)]
    for j in tqdm(range(n_hook_sides)):
        for i in range(j):
            l.append((i,j))
    
    random_order = np.random.choice(len(l),len(l),replace=False)
    
    d = {}   
    
    for n in tqdm(range(len(l))):
        (i, j) = l[random_order[n]]
        p0, p1 = hooks[i], hooks[j]
        d[(i,j)] = through_pixels(p0, p1)
    
    clear_output()
    return d