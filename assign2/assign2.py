import matplotlib.pyplot as plt
from networkx import nx
import operator as op
from functools import reduce
import random
import collections
import csv
import math

# Got this fucntion from https://stackoverflow.com/questions/4941753/is-there-a-math-ncr-function-in-python
def ncr(n, r):
    r = min(r, n-r)
    numer = reduce(op.mul, range(n, n-r, -1), 1)
    denom = reduce(op.mul, range(1, r+1), 1)
    return numer / denom

print("Generating the real world graph...")
realWolrdGraph = nx.DiGraph()
with open('collaboration_network.txt', 'r') as collaboration_network:
    firstLineRead = False
    for line in collaboration_network.readlines():
        if firstLineRead == False:
            firstLineRead = True
            continue
        
        edge = line.split('\t')
        realWolrdGraph.add_edge(int(edge[0]), int(edge[1]))

n = 5242
p = n/ncr(n, 2)

print("Generating ER graph...")
ER = nx.erdos_renyi_graph(n, p)

print("Generating small world graph...")
smallWorldGraph = nx.Graph()
for i in range(5242):
    smallWorldGraph.add_edge(i, (i-1)%5242)
    smallWorldGraph.add_edge(i, (i+1)%5242)
    smallWorldGraph.add_edge(i, (i-2)%5242)
    smallWorldGraph.add_edge(i, (i+2)%5242)

random_edges = random.sample(smallWorldGraph.edges, 4000)
for edge in random_edges:
    random.seed()
    random_node = random.randrange(5242)
    while random_node == edge[1]:
        random_node = random.randrange(5242)

    smallWorldGraph.remove_edge(edge[0], edge[1])
    smallWorldGraph.add_edge(edge[0], random_node)

realWorldDegrees = sorted([d for n, d in realWolrdGraph.degree()], reverse=True)
ERDegrees = sorted([d for n, d in ER.degree()], reverse=True)
smallWorldDegrees = sorted([d for n, d in smallWorldGraph.degree()], reverse=True)

realWorldCount = collections.Counter(realWorldDegrees)
ERCount = collections.Counter(ERDegrees)
smallWorldCount = collections.Counter(smallWorldDegrees)

RWdeg, RWcnt = zip(*realWorldCount.items())
ERdeg, ERcnt = zip(*ERCount.items())
SWdeg, SWcnt = zip(*smallWorldCount.items())

plt.style.use('classic')
plt.loglog(RWdeg, RWcnt, label='Real World Graph')
plt.loglog(ERdeg, ERcnt, label='ER Graph')
plt.loglog(SWdeg, SWcnt, label='Small World Graph')
plt.title('Degree Distrobutions')
plt.ylabel("Count")
plt.xlabel("Degree")
plt.legend()
plt.show()

print("Generating the Undirected real world graph...")
realWolrdGraph = nx.Graph()
with open('collaboration_network.txt', 'r') as collaboration_network:
    firstLineRead = False
    for line in collaboration_network.readlines():
        if firstLineRead == False:
            firstLineRead = True
            continue
        
        edge = line.split('\t')
        random.seed()
        trust = random.randrange(2)
        realWolrdGraph.add_edge(int(edge[0]), int(edge[1]), weight=trust)

triangleA = 0
triangleB = 0
triangleC = 0
triangleD = 0
print('Calculating number of triangles graph...')
for node in realWolrdGraph:
    for neighbor in realWolrdGraph.neighbors(node):
        if neighbor == node:
            continue
        for successor in realWolrdGraph.neighbors(neighbor):
            if successor == node:
                continue
            if successor in realWolrdGraph.neighbors(node):
                weights = collections.Counter([realWolrdGraph[node][neighbor]['weight'], realWolrdGraph[node][successor]['weight'], realWolrdGraph[neighbor][successor]['weight']])
                if weights[1] == 3:
                    triangleA += 1
                elif weights[0] == 1:
                    triangleB += 1
                elif weights[0] == 2:
                    triangleC += 1
                elif weights[0] == 3:
                    triangleD += 1

triangleA /= 3
triangleB /= 3
triangleC /= 3
triangleD /= 3

print("Count of Triangle Type A: " + str(triangleA))
print("Count of Triangle Type B: " + str(triangleB))
print("Count of Triangle Type C: " + str(triangleC))
print("Count of Triangle Type D: " + str(triangleD))

positveEdges = 0
negativeEdges = 0

for edge in realWolrdGraph.edges:
    weight = realWolrdGraph.get_edge_data(edge[0], edge[1])['weight']
    if weight == 1:
        positveEdges += 1
    else:
        negativeEdges += 1

print("Fraction of Positve edges: " + str(positveEdges) + '/' + str(realWolrdGraph.number_of_edges()))
print("Fraction of Negative edges: " + str(negativeEdges) + '/' + str(realWolrdGraph.number_of_edges()))

rp = positveEdges/realWolrdGraph.number_of_edges()
print("Probabilty of Triangle Type A: " + str(rp*rp*rp))
print("Probabilty of Triangle Type B: " + str((rp*rp*(1-rp)*3)))
print("Probabilty of Triangle Type C: " + str(rp*(1-rp)*(1-rp)*3))
print("Probabilty of Triangle Type D: " + str((1-rp)*(1-rp)*(1-rp)))

print("generating blog graph..")
blogGraph = nx.DiGraph()
with open('nodeslist.csv', 'r') as nodelist:
    nodelistreader = csv.DictReader(nodelist)
    for row in nodelistreader:
        blogGraph.add_node(row['\ufeffId'], URL=row['URL'], Label=row['Label'])

with open('edgelist.csv', 'r') as edgelist:
    edgelistreader = csv.DictReader(edgelist)
    for row in edgelistreader:
        blogGraph.add_edge(row['Source'], row['Target'])

print("predicting labels")
accuracy = dict()
number_of_nodes = blogGraph.number_of_nodes()
attributes = nx.get_node_attributes(blogGraph, 'Label')
nodes = blogGraph.nodes
for trial in range(10):
    for i in range(1,11):
        random.seed()
        nodeSample = random.sample(nodes, math.ceil(len(nodes)*(i/10)))
        correct = 0
        for node in blogGraph:
            neighbors = list(blogGraph.neighbors(node))
            liberalCout = 0
            consertiveCount = 0
            for neighbor in neighbors:
                if neighbor in nodeSample:
                    if  attributes[neighbor]== '0':
                        liberalCout += 1
                    else:
                        consertiveCount += 1

            if liberalCout >= consertiveCount:
                guess = '0'
            elif liberalCout < consertiveCount:
                guess = '1'
            else:
                print('uh oh')
            
            if guess == attributes[node]:
                correct += 1

        try:
            accuracy[i/10] += correct/number_of_nodes
        except:
            accuracy[i/10] = correct/number_of_nodes

for entry in accuracy:
    accuracy[entry] /= 10

plt.style.use('classic')
plt.plot(accuracy.keys(), accuracy.values())
plt.xlabel('Fraction of labels observed')
plt.ylabel('Accuracy')
plt.show()

print("Predicting Edges")
degreeAccuracy = dict()
jaccardAccuracy = dict()
adarAccuracy = dict()
blogNumberEdges = blogGraph.number_of_edges()
for trial in range(10):
    print("Trial: " + str(trial))
    degreeProducts = dict()
    jaccardIndex = dict()
    adarScore = dict()
    for i in range(1, 10):
        print("\tFraction: " + str(i/10))
        random.seed()
        edgeSample = random.sample(blogGraph.edges, math.ceil(blogNumberEdges*(i/10)))
        sampleGraph = nx.DiGraph()
        sampleGraph.add_edges_from(edgeSample)
        sampleNumberEdges = len(edgeSample)
        for u in blogGraph:
            try:
                gamma_u = set(sampleGraph.neighbors(u))
            except:
                gamma_u = set()

            for v in blogGraph:
                if u == v:
                    continue

                try:
                    degreeProducts[(u,v)] = len(list(sampleGraph.neighbors(u))) * len(list(sampleGraph.neighbors(v)))
                except:
                    degreeProducts[(u,v)] = 0
                
                try:
                    gamma_v = set(sampleGraph.neighbors(v))
                except:
                    gamma_v = set()

                try:
                    jaccardIndex[(u,v)] = len(gamma_u.intersection(gamma_v))/len(gamma_u.union(gamma_v))
                except:
                    jaccardIndex[(u,v)] = 0

                adarScore[(u,v)] = 0
                for z in gamma_u.intersection(gamma_v):
                    try:
                        adarScore[(u,v)] += 1/math.log(sampleGraph.degree(z))
                    except:
                        adarScore[(u,v)] += 0

        dd = collections.OrderedDict(sorted(degreeProducts.items(), key=lambda x: x[1], reverse=True))
        jj = collections.OrderedDict(sorted(jaccardIndex.items(), key=lambda x: x[1], reverse=True))
        aa = collections.OrderedDict(sorted(adarScore.items(), key=lambda x: x[1], reverse=True))

        degreeCorrect = 0
        jaccardCorrect = 0
        adarCorrect = 0
        n = blogNumberEdges - sampleNumberEdges
        if n == 0:
            degreeAccuracy[i/10] = 1
            jaccardAccuracy[i/10] = 1
            adarAccuracy[i/10] = 1
        else:
            ddKeys = list(dd.keys())
            jjkeys = list(jj.keys())
            aakeys = list(aa.keys())
            for j in range(n):
                if ddKeys[j] in blogGraph.edges:
                    degreeCorrect += 1
                if jjkeys[j] in blogGraph.edges:
                    jaccardCorrect += 1
                if aakeys[j] in blogGraph.edges:
                    adarCorrect += 1
            
            try:
                degreeAccuracy[i/10] += degreeCorrect/n
            except KeyError:
                degreeAccuracy[i/10] = degreeCorrect/n

            try:
                jaccardAccuracy[i/10] += jaccardCorrect/n
            except KeyError:
                jaccardAccuracy[i/10] = jaccardCorrect/n

            try:
                adarAccuracy[i/10] += adarCorrect/n
            except KeyError:
                adarAccuracy[i/10] = adarCorrect/n

for i in range(1, 10):
    degreeAccuracy[i/10] /= 10
    jaccardAccuracy[i/10] /= 10
    adarAccuracy[i/10] /= 10

plt.style.use('classic')
plt.plot(degreeAccuracy.keys(), degreeAccuracy.values(), label='Degree Product')
plt.plot(jaccardAccuracy.keys(), jaccardAccuracy.values(), label='Jaccard Index')
plt.plot(adarAccuracy.keys(), adarAccuracy.values(), label='Adar Score')
plt.xlabel('Fraction of edges observed')
plt.ylabel('Accuracy')
plt.legend()
plt.show()