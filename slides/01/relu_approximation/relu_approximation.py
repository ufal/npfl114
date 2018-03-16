#!/usr/bin/env python
from __future__ import division
from __future__ import print_function

import sys

import numpy as np
import tensorflow as tf

class Plotly():
    i = 0

    def start(self):
        print("<html>")
        print("<head><title>Graphs</title><script src='../../res/plotly/plotly-cartesian.min.js'></script></head>")
        print("<body>")
    def end(self):
        print("</body>")
        print("</html>")

    def plot_start(self):
        self.i += 1
        print("<div id='graph{}' style='width: 100%; height: 100%'/>".format(self.i))
        print("<script>")
        self.traces = []
    def plot_end(self, title, min_x, max_x):
        print("Plotly.newPlot('graph{}', [{}], {{title: '{}', xaxis: {{range: [{}, {}]}}}})".format(
            self.i, ",".join(self.traces), title, min_x, max_x))
        print("</script>")
    def plot_points(self, title, x, y, color, opacity, line_type, marker_color = None):
        self.traces.append("trace{}".format(len(self.traces)))
        print("{} = {{".format(self.traces[-1]))
        print("  type: 'scatter',")
        print("  name: '{}',".format(title))
        print("  opacity: {:.2f},".format(opacity))
        print("  mode: 'lines{}',".format("+markers" if marker_color else ""))
        print("  x: [{}],".format(",".join(map("{:.2f}".format, x))))
        print("  y: [{}],".format(",".join(map("{:.5f}".format, y))))
        if marker_color:
            print("  marker: {{size: {}, symbol: 'cross', color: '{}', line: {{color: '{}'}}}},".format(
                [0] + [10] * (len(x) - 2) + [0], marker_color, marker_color))
        print("  line: {{shape: '{}', color: '{}'}},".format(line_type, color))
        print("};")


MIN_X = -1
MAX_X = 1
GOLD_COEFS = [1, 0, -1.2, 0, 0.3, 0]

def data_compute(x):
    y = np.zeros_like(x)
    for coef in GOLD_COEFS:
        y = y * x + coef
    return y

def data_batch(batch_size):
    x = np.random.uniform(low=MIN_X, high=MAX_X, size=batch_size)
    return x, data_compute(x)


RELUS = 20

np.random.seed(42)
tf.set_random_seed(42)

session = tf.Session()

inputs = tf.placeholder(tf.float32, [None])
labels = tf.placeholder(tf.float32, [None])
hidden_w = tf.get_variable("hidden_weights", shape=[RELUS], dtype=tf.float32, initializer=tf.variance_scaling_initializer())
hidden_b = tf.get_variable("hidden_biases", shape=[RELUS], dtype=tf.float32, initializer=tf.variance_scaling_initializer())
hidden = tf.nn.relu(tf.matmul(tf.expand_dims(inputs, 1), tf.expand_dims(hidden_w, 0)) + hidden_b)
output_w = tf.get_variable("output_w", shape=[RELUS], dtype=tf.float32, initializer=tf.variance_scaling_initializer())
outputs = tf.squeeze(tf.matmul(hidden, tf.expand_dims(output_w, 1)), 1)

def relus_boundaries():
    h_w, h_b = session.run([hidden_w, hidden_b])

    boundaries = [MIN_X, MAX_X];
    for i in range(RELUS):
        x = -h_b[i] / h_w[i]
        if x > MIN_X and x < MAX_X:
            boundaries.append(x)
    return sorted(boundaries)

loss = tf.losses.mean_squared_error(labels, outputs)
train = tf.train.AdamOptimizer(0.005).minimize(loss)

session.run(tf.global_variables_initializer())

plotly = Plotly()
plotly.start()
plotly.plot_start()

EPOCHS = [0, 10, 100, 700, 1000, 8000]
OPACITIES = [0.35, 0.42, 0.5, 0.6, 0.7, 1]

l = None
for epoch in range(len(EPOCHS)):
    for _ in range(EPOCHS[epoch]):
        x, y = data_batch(257)
        o, l, _ = session.run([outputs, loss, train], {inputs: x, labels: y})

    print("Loss: {}".format(l), file=sys.stderr)
    plot_x = relus_boundaries()
    plotly.plot_points("Epoch {}".format(EPOCHS[epoch] if EPOCHS[epoch] < 1000 else str(EPOCHS[epoch]//1000) + "k"),
                       plot_x, session.run(outputs, {inputs: plot_x}), '#22b', OPACITIES[epoch], 'linear','#2b2')

PLOT_X = np.arange(MIN_X, MAX_X+0.01, 0.01)
plotly.plot_points("Original<br>Function", PLOT_X, data_compute(PLOT_X), '#e33', 1, 'spline')
plotly.plot_end("Evolving Approximation of a Polynomial Function by ReLUs", MIN_X, MAX_X)
plotly.end()
