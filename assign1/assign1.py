import networkx as nx
import matplotlib.pyplot as plt
import json
import numpy as np
from statistics import mean
import collections

def best_fit_slope_and_intercept(xs,ys):
    m = (((mean(xs)*mean(ys)) - mean(xs*ys)) /
         ((mean(xs)*mean(xs)) - mean(xs*xs)))
    
    b = mean(ys) - m*mean(xs)
    
    return m, b

graph_json = json.load(open('graph.json', 'r'))

print('Building dictonary Graph...')
graph = nx.DiGraph(graph_json)


print('Number of Nodes in the dictonary graph: ' + str(graph.number_of_nodes()))
print('Number of edges in the dictonary graph: ' + str(graph.number_of_edges()))

triangles = 0
print('Calculating number of triangles in the dictonary graph...')
for node in graph:
    for neighbor in graph.successors(node):
        if neighbor == node:
            continue
        for successor in graph.successors(neighbor):
            if successor == node:
                continue
            if successor in graph.predecessors(node):
                triangles += 1 


print('Number of Triangles in the dictonary graph: ' + str(triangles/3))

print('Calculating degrees of the dictonary graph...')
degrees = []
in_degrees = []
out_degrees = []
for node in graph_json.keys():
    degrees.append(graph.degree(node))
    out_degrees.append(graph.out_degree(node))
    in_degrees.append(graph.in_degree(node))

degree_counter = collections.Counter(degrees)
del degree_counter[0]
degree_counter_keys = [x for x in degree_counter.keys()]
degree_counter_values = [x for x in degree_counter.values()]

out_degree_counter = collections.Counter(out_degrees)
del out_degree_counter[0]
out_degree_counter_keys = [x for x in out_degree_counter.keys()]
out_degree_counter_values = [x for x in out_degree_counter.values()]

in_degree_counter = collections.Counter(in_degrees)
del in_degree_counter[0]
in_degree_counter_keys = [x for x in in_degree_counter.keys()]
in_degree_counter_values = [x for x in in_degree_counter.values()]

plt.loglog(degree_counter_keys, degree_counter_values, 'o', label='degree')
m, b = best_fit_slope_and_intercept(np.log10(np.array(degree_counter_keys)), np.log10(np.array(degree_counter_values)))
regression_line = [(10**b)*(x**m) for x in degree_counter_keys]
plt.loglog(degree_counter_keys, regression_line, label='a='+str(m)+' b='+str(b))

plt.loglog(out_degree_counter_keys, out_degree_counter_values, 'o', label='out degree')
m, b = best_fit_slope_and_intercept(np.log10(np.array(out_degree_counter_keys)), np.log10(np.array(out_degree_counter_values)))
regression_line = [(10**b)*(x**m) for x in out_degree_counter_keys]
plt.loglog(out_degree_counter_keys, regression_line, label='a='+str(m)+' b='+str(b))

plt.loglog(in_degree_counter_keys, in_degree_counter_values, 'o', label='in degree')
m, b = best_fit_slope_and_intercept(np.log10(np.array(in_degree_counter_keys)), np.log10(np.array(in_degree_counter_values)))
regression_line = [(10**b)*(x**m) for x in in_degree_counter_keys]
plt.loglog(in_degree_counter_keys, regression_line, label='a='+str(m)+' b='+str(b))

plt.xlabel('Degree')
plt.ylabel('Count')
plt.legend()
plt.ylim(10**0, 10**5)
plt.xlim(10**0, 10**5)
plt.show()

friendship_network = nx.Graph()

friendship_network_file = open('friendship_network.txt', 'r')
edges = []

print('Building friendship network...')
for line in friendship_network_file.readlines():
    line = line.replace('\n', '')
    line_list = line.split(' ')
    friendship_network.add_edge(int(line_list[0]), int(line_list[1]))

print('Computing number of connected componets in the friendship network...')
print('Number of connected componets: ' + str(nx.number_connected_components(friendship_network)))

print('Getting largest connected componet in the friendship network...')
lcc = max(nx.connected_components(friendship_network), key=len)
lcc = friendship_network.subgraph(lcc)

print('Number of nodes in the largest connected Componet of the firendship network: ' + str(lcc.number_of_nodes()))
print('Number of edges in the largest connected Componet of the firendship network: ' + str(lcc.number_of_edges()))

print('Computing the average neighbor degree of the friendship network...')
average_neighbor_degree = nx.average_neighbor_degree(friendship_network)

ratio = []
for node in friendship_network:
    ratio.append(friendship_network.degree[node]/average_neighbor_degree[node])

plt.scatter(average_neighbor_degree.values(), ratio)
plt.xlabel('Average Neighbor Degree')
plt.ylabel('Ratio')
plt.show()