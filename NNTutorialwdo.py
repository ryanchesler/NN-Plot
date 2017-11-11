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

global pause, layer_step, step_pause

#Define figure to plot to
fig = plt.figure(figsize=(12, 12))

#Set current position in animation to the start
layer_step = 0

#allow animation to auto iterate
step_pause = False

#Start the animation paused
pause = True


train_set = []
test_set = []



def label_data(location, shape, outlist, mode = "train"):
    files = os.listdir(location + str(shape))
    for file in files:
        if file[-3:] == "png":
            im = Image.open(location + str(shape) + "/" + file)
            if mode == "train":
              for x in range(3):
                im = im.rotate(90)
                imlist = list(np.array(im)[:,:,0].flatten())
                if mode == "train":
                  if shape == "/circles":
                      label = np.array([1, 0, 0, 0], ndmin=2).reshape((4,1))
                  elif shape == "/squares":
                      label = np.array([0, 1, 0, 0], ndmin=2).reshape((4,1))
                  elif shape == "/triangles":
                      label = np.array([0,0,1, 0], ndmin=2).reshape((4,1))
                out = [label] + [imlist]
                outlist.append(out)
            elif mode == "test":
              imlist = list(np.array(im)[:,:,0].flatten())
              label = [[]]
              out = [label] + [imlist]
              outlist.append(out)

def convert_to_input(train_set):
    data = np.array(train_set)
    train_y = data[0][0]
    train_x = data[0][1]
    for item in data[1:]:
        train_y = np.hstack((train_y, item[0]))
        train_x = np.vstack((train_x, item[1]))
    train_y = train_y.T
    input_shape = train_x.shape[1]
    output_shape = train_y.shape[1]
    return train_x, train_y, input_shape, output_shape



label_data("train", "/circles", train_set)
label_data("train", "/squares", train_set)
label_data("train", "/triangles", train_set)

label_data("test", "", test_set, "test")

def draw_neural_net(zs, weights, cost, acc = [1], left = .1, right = .9, bottom = .1, top = .9):
    '''zs: list of layers of list of nodes/edges. Format [[0.54,.34,.15], [.14,.27,.98]] would graph 2 layers with 3-nodes
       weights: list  of layers of list of weight matrix
       cost: list of recorded costs while training. This is used to compute the upper y bound for cost graph. You may manually pass in a different individual value
       acc : upper bound of accuracy. Default set to 1, but you can pass in your list of accuracies and it will make the upper bound your max value or you may assign manually
       left, right, bottom, top : margins'''
    global colors, p, pause, line1, line2, ax3, ax5
    #Colormap of Nodes
    maps = matplotlib.cm.autumn
    #gridspec layour
    gs = gridspec.GridSpec(16, 16)
    ax1 = plt.subplot(gs[0:4, 0:4])
    ax2 = plt.subplot(gs[5:9, 0:4])
    ax3 = plt.subplot(gs[10:14, 0:4])
    ax5 = plt.subplot(gs[:, 4:])
    line1, = ax1.plot([], [], lw=2)
    line2, = ax2.plot([], [], lw=2)
    ax1.set_ylabel('cost')
    ax1.set_xlabel('iterations')
    ax1.set_title("Cost")
    ax1.set_xlim(0,iterations)
    ax1.set_ylim(0,max(cost))

    ax2.set_ylabel('accuracy')
    ax2.set_xlabel('iterations')
    ax2.set_title("Accuracy %")
    ax2.set_xlim(0,iterations)
    ax2.set_ylim(0,max(acc))
    patches = []
    p = []
    nodes = []
    colors = []
    for node_count in zs:
        nodes.append(len(node_count))
    n_layers = len(nodes)
    v_spacing = (top - bottom)/float(max(nodes))
    h_spacing = (right - left)/float(len(nodes) - 1)
    ax5.axis('off')
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
                if m == 0:
                    text = "Circle"
                elif m == 1:
                    text = "Square"
                elif m == 2:
                    text = "Triangle"
                elif m == 3:
                    text = "I don't know"
                plt.text(n*h_spacing + left+0.10, layer_top - m*v_spacing, text, fontsize=12)
    p = PatchCollection(patches, cmap = maps, alpha =1)
    ax5.add_collection(p)
    
    # Edges
    for n, (layer_size_a, layer_size_b) in enumerate(zip(nodes[:-1], nodes[1:])):
        layer_top_a = v_spacing*(layer_size_a - 1)/2. + (top + bottom)/2.
        layer_top_b = v_spacing*(layer_size_b - 1)/2. + (top + bottom)/2.
        for m in range(layer_size_a):
            for o in range(layer_size_b):
                line = plt.Line2D([n*h_spacing + left, (n + 1)*h_spacing + left],
                              [layer_top_a - m*v_spacing, layer_top_b - o*v_spacing], c='k')
                ax5.add_artist(line)
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
def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.01)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.01, shape=shape)
  return tf.Variable(initial)

train_x, train_y, input_shape, output_shape = convert_to_input(train_set)
print(len(train_x))

hidden1 = 12
hidden2 = 12
hidden3 = 12

x = tf.placeholder(tf.float32, [None, input_shape])
W1 = weight_variable([input_shape, hidden1])
b1 = bias_variable([hidden1])
W2 = weight_variable([hidden1, hidden2])
b2 = bias_variable([hidden2])
W3 = weight_variable([hidden2, hidden3])
b3 = bias_variable([hidden3])
W4 = weight_variable([hidden3, output_shape])
b4 = bias_variable([output_shape])
y_ = tf.placeholder(tf.float32, [None, output_shape])
keep_prob1 = tf.placeholder(tf.float32)
keep_prob2 = tf.placeholder(tf.float32)
keep_prob3 = tf.placeholder(tf.float32)
drop1 = tf.nn.dropout(W1, keep_prob1)
drop2 = tf.nn.dropout(W2, keep_prob2)
drop3 = tf.nn.dropout(W3, keep_prob3)

layer1 = tf.nn.relu(tf.matmul(x, drop1) + b1)
layer2 = tf.nn.relu(tf.matmul(layer1, drop2) + b2)
layer3 = tf.nn.relu(tf.matmul(layer2, drop3) + b3)
layer4 = (tf.matmul(layer3, W4) + b4)

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=layer4))

train_step = tf.train.AdamOptimizer(0.001).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(layer4,1)+1, tf.argmax(y_,1)+1)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


acc, cost, zmac, wmac, pic_index, picture = [], [], [], [], [], []
iterations = 500
batch_size = -1
saver = tf.train.Saver()
with tf.Session() as sess:
    tf.global_variables_initializer().run()
    for i in range(iterations):
      list1_shuf, list2_shuf, weights, zs, pic_sample = [], [], [], [], []
      index_shuf = list(range(len(train_x)))
      shuffle(index_shuf)
      index_shuf = index_shuf[:batch_size]
      pic_index.append(index_shuf[0])
      for j in index_shuf:
        list1_shuf.append(train_x[j])
        list2_shuf.append(train_y[j])
      list_len = len(list1_shuf[0])
      part_len = list_len//10
      for part in range(10):
        pic_sample.append(np.average(list1_shuf[0][part_len*part:part_len*(part+1)])/255)
      _, accspot, costspot, weight1, weight2, weight3, weight4, lay1, lay2, lay3, lay4= sess.run([train_step, accuracy, cross_entropy, W1, W2, W3, W4, layer1, layer2, layer3, tf.nn.softmax(layer4)], feed_dict={x: list1_shuf, y_: list2_shuf, keep_prob1: .9, keep_prob2:.9, keep_prob3: .9})      
      acc.append(accspot)
      cost.append(costspot)
      weights.append(weight1.tolist())
      weights.append(weight2.tolist())
      weights.append(weight3.tolist())
      weights.append(weight4.tolist())
      picture.append(((list1_shuf[0])).tolist())
      zs.append(pic_sample)
      zs.append(lay1.tolist()[0])
      zs.append(lay2.tolist()[0])
      zs.append(lay3.tolist()[0])
      maxval = sess.run(tf.argmax(lay4[0]))
      if lay4[0][maxval] > .6:
        lay4[0][:] = 0
        lay4[0][maxval] = 1 
      else:
        lay4[0][:] = 0
        lay4[0][-1] = 1
      zs.append(lay4.tolist()[0])

      for index, value in enumerate(zs):
          xmin = min(value)
          xmax = max(value)
          xdif = xmax - xmin
          for subindex, subvalue in enumerate(value):
              if xdif == 0:
                  xdif = 1
              X_std = (subvalue - xmin) / xdif
              zs[index][subindex] = X_std
      zmac.append(zs)
      wmac.append(weights)
    saver.save(sess, "./checkpoint/model.ckpt")
print(acc[-1])          
draw_neural_net(zmac[0], wmac[0], cost, acc)
layer_step = 0

def onClick(event):
    global pause, step_pause
    pause ^= True
    step_pause = False
def on_key(event):
  global layer_step, step_pause, pause
  pause = False
  step_pause = True
  if event.key == "right":  
    layer_step += 1 
  if event.key == "left":  
    layer_step -= 1


def animate(i):
    global pause, p, layer_step, step_pause
    if not pause:
        colors = []
        current_step = 25*(layer_step//5)
        layer_step_point = layer_step % 5
        if current_step >= len(zmac):
            time.sleep(5)
            return line1, line2, ax3, p, ax5
        for layer_bound, layer in enumerate(zmac[current_step]):
            if layer_bound > layer_step_point:
                layer_colors = [0] * len(layer)
            else:
                layer_colors = layer
            colors.append(layer_colors)
        
        p.set_array(np.hstack(np.array(colors)))
        line1.set_data(range(current_step), cost[0:current_step])
        line2.set_data(range(current_step),acc[0:current_step])
        arr = np.array(picture[current_step]).reshape(50, 50)
        img = Image.fromarray(arr).convert("LA")
        ax3.imshow(img)
        if not step_pause:
          layer_step += 1
        ax5.set_title("Batch: " + str(current_step) + " Example: " + str(pic_index[current_step]))
        return line1, line2, ax3, p, ax5





cost = np.squeeze(cost)
acc = np.squeeze(acc)     
cid = fig.canvas.mpl_connect('button_press_event', onClick)
cid = fig.canvas.mpl_connect('key_press_event', on_key)
ani = animation.FuncAnimation(fig, animate, interval = 200)
plt.show()



test_x, test_y, input_shape, output_shape = convert_to_input(test_set)
zmac = []
picture = []
with tf.Session() as sess:
    saver.restore(sess, "./checkpoint/model.ckpt")
    for pic in test_x:
        weights, zs, pic_sample = [], [], []
        list_len = len(pic)
        part_len = list_len//10
        for part in range(10):
            pic_sample.append(np.average(pic[part_len*part:part_len*(part+1)])/255)
        
        picture.append(pic.tolist())
        zs.append(pic_sample)
        lay1, lay2, lay3, lay4 = sess.run([layer1, layer2, layer3, tf.nn.softmax(layer4)], feed_dict={x: [pic], keep_prob1: 1, keep_prob2:1, keep_prob3: 1})
        maxval = sess.run(tf.argmax(lay4[0]))
        if lay4[0][maxval] > .95:
          lay4[0][:] = 0
          lay4[0][maxval] = 1 
        else:
          lay4[0][:] = 0
          lay4[0][-1] = 1
        zs.append(lay1.tolist()[0])
        zs.append(lay2.tolist()[0])
        zs.append(lay3.tolist()[0])
        zs.append(lay4.tolist()[0])
        for index, value in enumerate(zs):
          xmin = min(value)
          xmax = max(value)
          xdif = xmax - xmin
          for subindex, subvalue in enumerate(value):
              if xdif == 0:
                  xdif = 1
              X_std = (subvalue - xmin) / xdif
              zs[index][subindex] = X_std
        zmac.append(zs)
fig = plt.figure(figsize= (12,12))
gs = gridspec.GridSpec(16, 16)
ax3 = plt.subplot(gs[10:14, 0:4])
ax5 = plt.subplot(gs[:, 4:])
patches = []
draw_neural_net(zmac[0], wmac[0], cost)
pause = True
layer_step = 0
def animate2(i):
    global pause
    if not pause:
        global p
        global layer_step
        global step_pause
        colors = []
        current_step = (layer_step//5)
        layer_step_point = layer_step % 5
        if current_step >= len(zmac):
            time.sleep(5)
            return ax3, p, ax5
        for layer_bound, layer in enumerate(zmac[current_step]):
            if layer_bound > layer_step_point:
                layer_colors = [0] * len(layer)
            else:
                layer_colors = layer
            colors.append(layer_colors)
        
        p.set_array(np.hstack(np.array(colors)))
        arr = np.array(picture[current_step]).reshape(50, 50)
        img = Image.fromarray(arr).convert("LA")
        ax3.imshow(img)
        if not step_pause:
          layer_step += 1
        ax5.set_title("Test: " + str(current_step))
        return ax3, p, ax5
cid = fig.canvas.mpl_connect('button_press_event', onClick)
cid = fig.canvas.mpl_connect('key_press_event', on_key)
ani = animation.FuncAnimation(fig, animate2)
plt.show()


