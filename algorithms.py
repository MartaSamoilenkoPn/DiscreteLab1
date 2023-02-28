import random
import networkx as nx
import matplotlib.pyplot as plt
from itertools import combinations, groupby

# You can use this function to generate a random graph with 'num_of_nodes' nodes
# and 'completeness' probability of an edge between any two nodes
# If 'directed' is True, the graph will be directed
# If 'draw' is True, the graph will be drawn
def gnp_random_connected_graph(num_of_nodes: int,
                               completeness: int,
                               directed: bool = False,
                               draw: bool = False):
    """
    Generates a random graph, similarly to an Erdős-Rényi 
    graph, but enforcing that the resulting graph is conneted (in case of undirected graphs)
    """

    if directed:
        G = nx.DiGraph()
    else:
        G = nx.Graph()
    edges = combinations(range(num_of_nodes), 2)
    G.add_nodes_from(range(num_of_nodes))

    for _, node_edges in groupby(edges, key = lambda x: x[0]):
        node_edges = list(node_edges)
        random_edge = random.choice(node_edges)
        if random.random() < 0.5:
            random_edge = random_edge[::-1]
        G.add_edge(*random_edge)
        for e in node_edges:
            if random.random() < completeness:
                G.add_edge(*e)

    for (u,v,w) in G.edges(data=True):
        w['weight'] = random.randint(-5, 20)

    if draw: 
        plt.figure(figsize=(10,6))
        if directed:
            # draw with edge weights
            pos = nx.arf_layout(G)
            nx.draw(G,pos, node_color='lightblue', 
                    with_labels=True,
                    node_size=500, 
                    arrowsize=20, 
                    arrows=True)
            labels = nx.get_edge_attributes(G,'weight')
            nx.draw_networkx_edge_labels(G, pos,edge_labels=labels)
        else:
            nx.draw(G, node_color='lightblue', 
                with_labels=True, 
                node_size=500)
    return G

#kruskal
G = gnp_random_connected_graph(10, 0.5, False, True)
print(G.edges())

def find_set(vertex_list, vertex):
    if vertex_list[vertex] == vertex:
        return vertex
    return find_set(vertex_list, vertex_list[vertex])

def union(vertex_list, deg_list, vertex_u, vertex_v):
    find_u = find_set(vertex_list, vertex_u)
    find_v = find_set(vertex_list, vertex_v)
    if deg_list[find_u] < deg_list[find_v]:
        vertex_list[find_u] = find_v
    elif deg_list[find_v] < deg_list[find_u]:
        vertex_list[find_v] = find_u
    else:
        vertex_list[find_v] = find_u
        deg_list[find_u] += 1
    return vertex_list, deg_list

def kruskal_algorithm(G):
    result = []
    vertex_list = []
    deg_list = []
    for vertex in G.nodes():
        vertex_list.append(vertex)
        deg_list.append(0)

    graph_start_list = []

    for edge in G.edges():
        graph_start_list.append([edge[0], edge[1], G.get_edge_data(edge[0], edge[1])['weight']])

    graph_list = list(sorted(graph_start_list, key = lambda x : x[2]))

    for edge in graph_list:
        u = edge[0]
        v = edge[1]
        if find_set(vertex_list, u) != find_set(vertex_list ,v):
            result.append([u, v, edge[2]])
            vertex_list, deg_list = union(vertex_list, deg_list, u, v)

    return result

print("kruskal edges")
for element in kruskal_algorithm(G):
    print(element[0], element[1])

#bellman_ford
G = gnp_random_connected_graph(10, 0.5, True, True)

def relax(vertex_u, vertex_v, weight, distance_dict, pi_dict):
    if distance_dict[vertex_v] > distance_dict[vertex_u] + weight:
        distance_dict[vertex_v] = distance_dict[vertex_u] + weight
        pi_dict[vertex_v] = vertex_u

    return distance_dict, pi_dict

def bellman_ford(G, s):
    distance_dict = {}
    pi_dict = {}
    for vertex in G.nodes():
        distance_dict[vertex] = float('inf')
        pi_dict[vertex] = 0
    distance_dict[s] = 0

    graph_dict = {}

    for node_start in range(len(G.nodes())):
        for node_finish in range(len(G.nodes())):
            if G.get_edge_data(node_start, node_finish) != None:
                graph_dict[(node_start, node_finish)] = G.get_edge_data(node_start, node_finish)['weight']

    for _ in range(len(G.edges())-1):
        for edge in G.edges():
            distance_dict, pi_dict = relax(edge[0], edge[1], graph_dict[edge], distance_dict, pi_dict)

    for edge in G.edges():
        if distance_dict[edge[1]] > distance_dict[edge[0]] + graph_dict[edge]:
            return False

    return distance_dict

print()
for edge in G.edges:
    print(edge, G.get_edge_data(edge[0], edge[1])['weight'])
print("distances by bellman ford")
distance = bellman_ford(G, 0)
for key in distance:
    if distance[key] != float('inf'):
        print(f"Distance to {key} = {distance[key]}")
