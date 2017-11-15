import matplotlib.colors as clrs
import tensorflow as tf
import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.colors as clrs
import matplotlib
import matplotlib.gridspec as gridspec
from matplotlib import animation
from matplotlib.collections import PatchCollection
import random
from random import shuffle
import time
from tkinter import filedialog

def draw_neural_net(nn, zs, weights, labels = [], left = .1, right = .9, bottom = .1, top = .9):
    p = []
    nodes = []
    patches = []
    colors = []
    maps = matplotlib.cm.autumn
    nn.axis('off')
    for node_count in zs:
        nodes.append(len(node_count))
    n_layers = len(nodes)
    v_spacing = (top - bottom)/float(max(nodes))
    h_spacing = (right - left)/float(len(nodes) - 1)
        #input-arrows
    layer_top_0 = v_spacing*(nodes[0] - 1)/2. + (top + bottom)/2.
    for m in range(nodes[0]):
        plt.arrow(left-0.18, layer_top_0 - m*v_spacing, 0.12, 0,  lw =1, head_width=0.01, head_length=0.02)
    # Nodes
    for n, layer_size in enumerate(nodes):
        layer_top = v_spacing*(layer_size - 1)/2. + (top + bottom)/2.
        for m in range(layer_size):
            colors.append(1-zs[n][m])
            x = n*h_spacing + left
            y = layer_top - m*v_spacing
            circle = plt.Circle((x,y), v_spacing/4., ec='k', zorder=5)
            patches.append(circle)
            if n == 0:
                plt.text(left-0.125, layer_top - m*v_spacing, r'$X_{'+str(m+1)+'}$', fontsize=15)
            elif n == n_layers -1:
                text = labels[m]
                plt.text(n*h_spacing + left+0.10, layer_top - m*v_spacing, text, fontsize=12)
    p = PatchCollection(patches, cmap = maps, alpha =1)
    nn.add_collection(p)
    
    # Edges
    for n, (layer_size_a, layer_size_b) in enumerate(zip(nodes[:-1], nodes[1:])):
        layer_top_a = v_spacing*(layer_size_a - 1)/2. + (top + bottom)/2.
        layer_top_b = v_spacing*(layer_size_b - 1)/2. + (top + bottom)/2.
        for m in range(layer_size_a):
            for o in range(layer_size_b):
                line = plt.Line2D([n*h_spacing + left, (n + 1)*h_spacing + left],
                              [layer_top_a - m*v_spacing, layer_top_b - o*v_spacing], c='k')
                nn.add_artist(line)
                xm = (n*h_spacing + left)
                xo = ((n + 1)*h_spacing + left)
                ym = (layer_top_a - m*v_spacing)
                yo = (layer_top_b - o*v_spacing)
                rot_mo_rad = np.arctan((yo-ym)/(xo-xm))
                rot_mo_deg = rot_mo_rad*180./np.pi
                xm1 = xm + (v_spacing/8.+0.05)*np.cos(rot_mo_rad)
                if n == 0:
                    if yo > ym:
                        ym1 = ym + (v_spacing/8.+0.12)*np.sin(rot_mo_rad)
                    else:
                        ym1 = ym + (v_spacing/8.+0.05)*np.sin(rot_mo_rad)
                else:
                    if yo > ym:
                        ym1 = ym + (v_spacing/8.+0.12)*np.sin(rot_mo_rad)
                    else:
                        ym1 = ym + (v_spacing/8.+0.04)*np.sin(rot_mo_rad)
                plt.text( xm1, ym1, str(round(weights[n][m][o],4)), rotation = rot_mo_deg, fontsize = 10)
    return p
def draw_cost(cost_plot, cost):
    line1, = cost_plot.plot([], [], lw=2)
    cost_plot.set_ylabel('cost')
    cost_plot.set_xlabel('iterations')
    cost_plot.set_title("Cost")
    cost_plot.set_xlim(0,len(cost))
    cost_plot.set_ylim(0,max(cost))
    return line1
def draw_accuracy(accuracy_plot, acc):
    line2, = accuracy_plot.plot([], [], lw=2)
    accuracy_plot.set_ylabel('accuracy')
    accuracy_plot.set_xlabel('iterations')
    accuracy_plot.set_title("Accuracy %")
    accuracy_plot.set_xlim(0,len(acc))
    accuracy_plot.set_ylim(0,max(acc))
    return line2
##fig = plt.figure(figsize=(12,12))
##gs = gridspec.GridSpec(16,16)
##ax1 = plt.subplot(gs[:, 4:])
##cost = [0,1,2,3,4,5]
##accuracy = [0,1,2,3,4,5]
##draw_neural_net(ax1, [[12,32,12],[3,4,5]],[[[1,32,12], [34,56,13],[12,76,41]]], ["Circle", "Square", "Triangle", "I Don't Know"])
##ax2 = plt.subplot(gs[0:4, 0:4])
##draw_cost(ax2, cost)
##ax3 = plt.subplot(gs[5:9, 0:4])
##draw_accuracy(ax3, accuracy)
##plt.show()
