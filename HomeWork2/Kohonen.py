import SimpSOM as sps
import numpy as np
import math

colors = math.sqrt(255) * np.random.randn(1600,3)
labels = np.zeros([1600,3])

net = sps.somNet(40, 40, colors, PBC=True)

net.train(0.05, 1000)

net.save('filename_weights')
net.nodes_graph(colnum=0)
net.diff_graph()
net.project(colors, labels=labels)
net.cluster(colors, type='qthresh')

