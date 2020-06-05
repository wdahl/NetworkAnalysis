import networkx as nx
from datetime import datetime
import matplotlib.pyplot as plt
import collections
import numpy as np
import random

graph = nx.DiGraph()
fourWeekGraph = nx.DiGraph()
eightWeekGraph = nx.DiGraph()

componetRatios = list()
fourWeekRatios = list()
eightWeekRatios = list()

powerExp = list()

geodesicDistances = list()

days = list()
dayCount = 1
with open('temporalnetwork.csv', 'r') as network:
    day = None
    header = True
    for line in network.readlines():
        if header:
            header = False
            continue
        
        line = line.rstrip()
        line = line.split(',')
        ts = int(line[2])
        newDay = datetime.utcfromtimestamp(ts).strftime('%d')

        if day == None:
            day = newDay

        if day == newDay:
            graph.add_edge(int(line[0]), int(line[1]))
            fourWeekGraph.add_edge(int(line[0]), int(line[1]), weight=28)
            eightWeekGraph.add_edge(int(line[0]), int(line[1]), weight=56)
        
        else:
            wcc = max(nx.weakly_connected_components(graph), key=len)
            componetRatios.append(len(wcc)/graph.number_of_nodes())
            fourWeekRatios.append(len(max(nx.weakly_connected_components(fourWeekGraph), key=len))/fourWeekGraph.number_of_nodes())
            eightWeekRatios.append(len(max(nx.weakly_connected_components(eightWeekGraph), key=len))/eightWeekGraph.number_of_nodes())

            days.append(dayCount)
            dayCount += 1
            day = None

            edges = list()
            for edge in fourWeekGraph.edges:
                if fourWeekGraph[edge[0]][edge[1]]['weight'] == 1:
                    edges.append(edge)
                else:
                    fourWeekGraph[edge[0]][edge[1]]['weight'] -= 1

            for edge in edges:
                fourWeekGraph.remove_edge(edge[0], edge[1])

            edges = list()
            for edge in eightWeekGraph.edges:
                if eightWeekGraph[edge[0]][edge[1]]['weight'] == 1:
                    edges.append(edge)
                else:
                    eightWeekGraph[edge[0]][edge[1]]['weight'] -= 1

            for edge in edges:
                eightWeekGraph.remove_edge(edge[0], edge[1])

            degree = graph.degree()
            degree_sequence = sorted([d for n, d in degree], reverse=True) # sorts the degrees in descinding order
            degreeCount = collections.Counter(degree_sequence) # counts the number of times a degree occurs
            deg, cnt = zip(*degreeCount.items()) # gets the list of the degrees and the corisponding count
            
            if(len(degreeCount) == 1):
                powerExp.append(0)
            else:
                m, c = np.polyfit(np.log(deg), np.log(cnt), 1)
                powerExp.append(m)

            '''    
            try:
                geodesicDistances.append(nx.average_shortest_path_length(graph))
            except:
                geodesicDistances.append(0)
            '''
            wcc_g = graph.subgraph(wcc)
            geodesicDistances.append(nx.average_shortest_path_length(wcc_g))
            graph.add_edge(int(line[0]), int(line[1]))
            fourWeekGraph.add_edge(int(line[0]), int(line[1]), weight=28)
            eightWeekGraph.add_edge(int(line[0]), int(line[1]), weight=56)

plt.plot(days, componetRatios, label='Indefinite lifespan')
plt.plot(days, fourWeekRatios, label='4 week lifespan')
plt.plot(days, eightWeekRatios, label='8 week lifespan')
plt.xlabel('Day')
plt.ylabel('Component Size')
plt.legend()
plt.show()

plt.plot(days, powerExp)
plt.xlabel('Days')
plt.ylabel('Power Law Exponent')
plt.show()

plt.plot(days, geodesicDistances)
plt.xlabel('Days')
plt.ylabel('Average Geodesic Distance')
plt.show()

graph = nx.Graph()
with open('interactionsnetwork.csv', 'r') as network:
    header = True
    for line in network.readlines():
        if header:
            header = False
            continue

        line = line.rstrip()
        line = line.split(',')

        graph.add_node(int(line[0]), inf=0)
        graph.add_node(int(line[1]), inf=0)
        graph.add_edge(int(line[0]), int(line[1]))

randomNet1 = nx.Graph()
with open('RandNet1.csv', 'r') as network:
    header = True
    for line in network.readlines():
        if header:
            header = False
            continue

        line = line.rstrip()
        line = line.split(',')

        randomNet1.add_node(int(line[0]), inf=0)
        randomNet1.add_node(int(line[1]), inf=0)
        randomNet1.add_edge(int(line[0]), int(line[1]))


randomNet2 = nx.Graph()
with open('RandNet2.csv', 'r') as network:
    header = True
    for line in network.readlines():
        if header:
            header = False
            continue

        line = line.rstrip()
        line = line.split(',')

        randomNet2.add_node(int(line[0]), inf=0)
        randomNet2.add_node(int(line[1]), inf=0)
        randomNet2.add_edge(int(line[0]), int(line[1]))

def infect(G, intial_infected):
    curr_infected = [intial_infected]
    G.nodes[intial_infected]['inf'] = 1
    new_infected = list()

    t = 0
    infected_count = 1
    while len(curr_infected) != 0:
        t += 1
        for node in curr_infected:
            for neighbor in G.neighbors(node):
                if G.nodes[neighbor]['inf'] != 1:
                    choice = random.sample(range(1, 11), 1)[0]
                    if choice <= i:
                        G.nodes[neighbor]['inf'] = 1
                        infected_count += 1
                        new_infected.append(neighbor)

        curr_infected = new_infected
        new_infected = list()    

    return infected_count, t

graph_node_count = graph.number_of_nodes()
randNet1_node_count = randomNet1.number_of_nodes()
randNet2_node_count = randomNet2.number_of_nodes()

graph_sizes = dict()
graph_lens = dict()

randNet1_sizes = dict()
randNet1_lens = dict()

randNet2_sizes = dict()
randNet2_lens = dict()
for trial in range(100):
    print(trial)

    graph_intial = random.sample(graph.nodes, 1)[0]   
    randNet1_intial = random.sample(randomNet1.nodes, 1)[0]
    randNet2_intial = random.sample(randomNet2.nodes, 1)[0]
    for i in range(11):
        p = i/10

        size, t = infect(graph, graph_intial)
        try:
            graph_sizes[p] += (size/graph_node_count)
        except:
            graph_sizes[p] = (size/graph_node_count)

        try:
            graph_lens[p] += t
        except:
            graph_lens[p] = t

        nx.set_node_attributes(graph, 0, 'inf')

        size, t = infect(randomNet1, randNet1_intial)
        try:
            randNet1_sizes[p] += (size/randNet1_node_count)
        except:
            randNet1_sizes[p] = (size/randNet1_node_count)

        try:
            randNet1_lens[p] += t
        except:
            randNet1_lens[p] = t
        
        nx.set_node_attributes(randomNet1, 0, 'inf')

        size, t = infect(randomNet2, randNet2_intial)
        try:
            randNet2_sizes[p] += (size/randNet2_node_count)
        except:
            randNet2_sizes[p] = (size/randNet2_node_count)

        try:
            randNet2_lens[p] += t
        except:
            randNet2_lens[p] = t

        nx.set_node_attributes(randomNet2, 0, 'inf')

for p in graph_sizes:
    graph_sizes[p] /= 100
    graph_lens[p] /= 100

    randNet1_sizes[p] /= 100
    randNet1_lens[p] /= 100

    randNet2_sizes[p] /= 100
    randNet2_lens[p] /= 100

plt.plot(graph_sizes.keys(), graph_sizes.values(), label='Interaction Network')
plt.plot(randNet1_sizes.keys(), randNet1_sizes.values(), label='Random Network 1')
plt.plot(randNet2_sizes.keys(), randNet2_sizes.values(), label='Random Network 2')
plt.vlines(0.2, 0, 1, label='Critical Value')
plt.xlabel('Probabilty of Infection')
plt.ylabel('Size of Epidemic')
plt.legend()
plt.show()

plt.plot(graph_lens.keys(), graph_lens.values(), label='Interaction Network')
plt.plot(randNet1_lens.keys(), randNet1_lens.values(), label='Random Network 1')
plt.plot(randNet2_lens.keys(), randNet2_lens.values(), label='Random Network 2')
plt.vlines(0.2, 0, 20, label='Critical Value')
plt.hlines(np.log(graph_node_count), 0, 1, label='log n')
plt.xlabel('Probabilty of Infection')
plt.ylabel('Length of Epidemic')
plt.legend()
plt.show()

def get_degree(degree_tuple):
    return degree_tuple[1]

graph_intial = max(graph.degree, key=get_degree)[1]
randNet1_intial = max(randomNet1.degree, key=get_degree)[1]
randNet2_intial = max(randomNet2.degree, key=get_degree)[1]

graph_sizes_d = dict()
graph_lens_d = dict()

randNet1_sizes_d = dict()
randNet1_lens_d = dict()

randNet2_sizes_d = dict()
randNet2_lens_d = dict()
for trial in range(100):
    print(trial)

    for i in range(11):
        p = i/10

        size, t = infect(graph, graph_intial)
        try:
            graph_sizes_d[p] += (size/graph_node_count)
        except:
            graph_sizes_d[p] = (size/graph_node_count)

        try:
            graph_lens_d[p] += t
        except:
            graph_lens_d[p] = t

        nx.set_node_attributes(graph, 0, 'inf')

        size, t = infect(randomNet1, randNet1_intial)
        try:
            randNet1_sizes_d[p] += (size/randNet1_node_count)
        except:
            randNet1_sizes_d[p] = (size/randNet1_node_count)

        try:
            randNet1_lens_d[p] += t
        except:
            randNet1_lens_d[p] = t
        
        nx.set_node_attributes(randomNet1, 0, 'inf')

        size, t = infect(randomNet2, randNet2_intial)
        try:
            randNet2_sizes_d[p] += (size/randNet2_node_count)
        except:
            randNet2_sizes_d[p] = (size/randNet2_node_count)

        try:
            randNet2_lens_d[p] += t
        except:
            randNet2_lens_d[p] = t

        nx.set_node_attributes(randomNet2, 0, 'inf')

for p in graph_sizes_d:
    graph_sizes_d[p] /= 100
    graph_lens_d[p] /= 100

    randNet1_sizes_d[p] /= 100
    randNet1_lens_d[p] /= 100

    randNet2_sizes_d[p] /= 100
    randNet2_lens_d[p] /= 100

plt.plot(graph_sizes_d.keys(), graph_sizes_d.values(), label='Interaction Network')
plt.plot(randNet1_sizes_d.keys(), randNet1_sizes_d.values(), label='Random Network 1')
plt.plot(randNet2_sizes_d.keys(), randNet2_sizes_d.values(), label='Random Network 2')
plt.vlines(0.3, 0, 1, label='Critical Value')
plt.xlabel('Probabilty of Infection')
plt.ylabel('Size of Epidemic')
plt.legend()
plt.show()

plt.plot(graph_lens_d.keys(), graph_lens_d.values(), label='Interaction Network')
plt.plot(randNet1_lens_d.keys(), randNet1_lens_d.values(), label='Random Network 1')
plt.plot(randNet2_lens_d.keys(), randNet2_lens_d.values(), label='Random Network 2')
plt.vlines(0.3, 0, 20, label='Critical Value')
plt.hlines(np.log(graph_node_count), 0, 1, label='log n')
plt.xlabel('Probabilty of Infection')
plt.ylabel('Length of Epidemic')
plt.legend()
plt.show()

print('Relative increase in epidmic size for Interaction Netwrok: ', sum(graph_sizes_d.values()) - sum(graph_sizes.values()))
print('Relative increase in epidmic size for Random Network 1: ', sum(randNet1_sizes_d.values()) - sum(randNet1_sizes.values()))
print('Relative increase in epidmic size for Random Network 2: ', sum(randNet2_sizes_d.values()) - sum(randNet2_sizes.values()))