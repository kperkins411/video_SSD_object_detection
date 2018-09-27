import collections
import numpy as np
from matplotlib import patches, patheffects
from fastai.dataset import *

import matplotlib.cm as cmx
import matplotlib.colors as mcolors
from cycler import cycler


#default dict for each image ID, holds bounding box and class Ids for each image
# 12  <class 'list'>: [(array([ 96, 155, 269, 350]), 7)]        #1 item
# 17 <class 'list'>: [(array([ 61, 184, 198, 278]), 15), (array([ 77,  89, 335, 402]), 13)] #2 items
def get_trn_anno(*,trn_j,ano,bbox, img_ID, cat_ID):
    trn_anno = collections.defaultdict(lambda: [])
    for o in trn_j[ano]:
        if not o['ignore']:
            bb = o[bbox]
            bb = np.array([bb[1], bb[0], bb[3] + bb[1] - 1, bb[2] + bb[0] - 1])
            trn_anno[o[img_ID]].append((bb, o[cat_ID]))
    return trn_anno

def show_img(im, figsize=None, ax=None):
    if not ax: fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(im)
    ax.set_xticks(np.linspace(0, 224, 8))
    ax.set_yticks(np.linspace(0, 224, 8))
    ax.grid()
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    return ax


def draw_outline(o, lw):
    o.set_path_effects([patheffects.Stroke(
        linewidth=lw, foreground='black'), patheffects.Normal()])


def draw_rect(ax, b, color='white'):
    patch = ax.add_patch(patches.Rectangle(b[:2], *b[-2:], fill=False, edgecolor=color, lw=2))
    draw_outline(patch, 4)


def draw_text(ax, xy, txt, sz=14, color='white'):
    text = ax.text(*xy, txt,
                   verticalalignment='top', color=color, fontsize=sz, weight='bold')
    draw_outline(text, 1)

def bb_hw(a):
    return np.array([a[1], a[0], a[3] - a[1] + 1, a[2] - a[0] + 1])

def draw_im(im, ann, cats):
    ax = show_img(im, figsize=(16, 8))
    for b, c in ann:
        b = bb_hw(b)
        draw_rect(ax, b)
        draw_text(ax, b[:2], cats[c], sz=16)

def draw_idx(i, trn_anno, pth, trn_fns, cats):
    im_a = trn_anno[i]
    # im = open_image(IMG_PATH / trn_fns[i])
    im = open_image(pth / trn_fns[i])
    draw_im(im, im_a, cats)


def get_cmap(N):
    color_norm = mcolors.Normalize(vmin=0, vmax=N - 1)
    return cmx.ScalarMappable(norm=color_norm, cmap='Set3').to_rgba

def show_ground_truth(ax, im, bbox,id2cat1, clas=None, prs=None, thresh=0.3):
    num_colr = 12
    cmap = get_cmap(num_colr)
    colr_list = [cmap(float(x)) for x in range(num_colr)]

    bb = [bb_hw(o) for o in bbox.reshape(-1, 4)]
    if prs is None:  prs = [None] * len(bb)
    if clas is None: clas = [None] * len(bb)
    ax = show_img(im, ax=ax)
    for i, (b, c, pr) in enumerate(zip(bb, clas, prs)):
        if ((b[2] > 0) and (pr is None or pr > thresh)):
            draw_rect(ax, b, color=colr_list[i % num_colr])
            txt = f'{i}: '
            if c is not None: txt += ('bg' if c == len(id2cat1) else id2cat1[c])
            if pr is not None: txt += f' {pr:.2f}'
            draw_text(ax, b[:2], txt, color=colr_list[i % num_colr])

