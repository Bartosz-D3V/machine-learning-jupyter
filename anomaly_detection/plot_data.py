def plot_data(pyplot, x):
    pyplot.plot(x[:, 0], x[:, 1], 'kx')
    pyplot.xlabel('Latency (ms)')
    pyplot.ylabel('Throughput (mb/s)')
