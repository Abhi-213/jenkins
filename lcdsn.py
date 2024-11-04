import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from community import community_louvain
from scipy.io import mmread
from sklearn.metrics import normalized_mutual_info_score

# Load the dataset
def load_graph_from_mtx(filename):
    matrix = mmread(filename).tocoo()
    graph = nx.Graph()
    for i, j, value in zip(matrix.row, matrix.col, matrix.data):
        graph.add_edge(i, j, weight=value)
    return graph

def calculate_imp(graph, alpha=0.7, beta=0.3, gamma=6):
    imp = {node: 1.0 for node in graph.nodes}
    for _ in range(gamma):
        new_imp = {}
        for node in graph.nodes:
            in_neighbors = list(graph.neighbors(node))
            second_neighbors = set()

            # Collect second-degree neighbors
            for neighbor in in_neighbors:
                second_neighbors.update(graph.neighbors(neighbor))

            imp_value = 0
            
            # Contribution from first-degree neighbors
            for neighbor in in_neighbors:
                if graph.has_edge(node, neighbor):
                    weight = graph[node][neighbor].get('weight', 1)
                    out_weight_sum = sum(graph[neighbor][n].get('weight', 1) for n in graph.neighbors(neighbor))
                    if out_weight_sum > 0:
                        imp_value += (alpha * weight * imp[neighbor]) / out_weight_sum

            # Contribution from second-degree neighbors
            for second_neighbor in second_neighbors:
                if second_neighbor != node and graph.has_edge(node, second_neighbor):
                    weight = graph[node][second_neighbor].get('weight', 1)
                    out_weight_sum = sum(graph[second_neighbor][n].get('weight', 1) for n in graph.neighbors(second_neighbor))
                    if out_weight_sum > 0:
                        imp_value += (beta * weight * imp[second_neighbor]) / out_weight_sum

            new_imp[node] = imp_value
        
        imp = new_imp
    return imp


def form_initial_communities(graph, imp):
    sorted_nodes = sorted(imp, key=imp.get, reverse=True)
    communities = []
    assigned = set()

    for node in sorted_nodes:
        if node not in assigned:
            community = {node}
            community.update(graph.neighbors(node))
            communities.append(community)
            assigned.update(community)

    print(f"Node importance: {imp}")
    print(f"Initial communities formed: {communities}")
    return communities

def glhn_similarity(graph, node1, node2):
    neighbors1 = set(graph.neighbors(node1))
    neighbors2 = set(graph.neighbors(node2))
    shared_neighbors = neighbors1.intersection(neighbors2)
    return len(shared_neighbors) / (len(neighbors1) * len(neighbors2)) ** 0.5 if len(neighbors1) and len(neighbors2) else 0

def calculate_similarity(graph, node, community):
    return sum(glhn_similarity(graph, node, member) for member in community)

def resolve_overlaps(graph, communities):
    final_communities = []
    for community in communities:
        overlaps = set()
        for node in community:
            for other_community in communities:
                if node in other_community and community != other_community:
                    overlaps.add(node)
        
        for overlap_node in overlaps:
            best_community = max(communities, key=lambda c: calculate_similarity(graph, overlap_node, c))
            if best_community != community:
                community.remove(overlap_node)
        
        final_communities.append(community)

    print(f"Communities after resolving overlaps: {final_communities}")
    return final_communities

def merge_small_weak_communities(graph, communities, mc=4):
    merged_communities = []
    for community in communities:
        internal_edges = sum(1 for node in community for neighbor in graph.neighbors(node) if neighbor in community)
        external_edges = sum(1 for node in community for neighbor in graph.neighbors(node) if neighbor not in community)
        
        # Merge if internal connections are weak
        if internal_edges <= mc * external_edges:
            best_community = max(communities, key=lambda c: calculate_similarity(graph, list(community)[0], c))
            if best_community not in merged_communities:
                merged_communities.append(best_community)
        else:
            merged_communities.append(community)

    # Remove duplicate communities
    unique_communities = []
    for community in merged_communities:
        if community not in unique_communities:
            unique_communities.append(community)

    # Handle isolated nodes or small communities
    isolated_nodes = set(graph.nodes) - set(node for community in unique_communities for node in community)
    for isolated_node in isolated_nodes:
        best_community = max(unique_communities, key=lambda c: calculate_similarity(graph, isolated_node, c))
        best_community.add(isolated_node)

    print(f"Communities after merging: {unique_communities}")
    return unique_communities

def calculate_modularity(graph, communities):
    partition = {}
    for idx, community in enumerate(communities):
        for node in community:
            partition[node] = idx
    
    for node in graph.nodes:
        if node not in partition:
            partition[node] = -1
    
    modularity = community_louvain.modularity(partition, graph)
    return modularity

def calculate_nmi(ground_truth, predicted):
    return normalized_mutual_info_score(ground_truth, predicted)

def lcd_sn_algorithm(graph, ground_truth_labels, alpha=0.7, beta=0.3, gamma=6, mc=4):
    imp = calculate_imp(graph, alpha, beta, gamma)
    communities = form_initial_communities(graph, imp)
    communities = resolve_overlaps(graph, communities)
    final_communities = merge_small_weak_communities(graph, communities, mc)
    modularity_value = calculate_modularity(graph, final_communities)

    # Flatten final communities for NMI calculation
    predicted_labels = [-1] * len(graph.nodes)
    for idx, community in enumerate(final_communities):
        for node in community:
            if node < len(predicted_labels):  # Ensure node exists
                predicted_labels[node] = idx
    
    # Ensure predicted_labels has the same length as ground_truth_labels
    predicted_labels = predicted_labels[:len(ground_truth_labels)]

    # Ensure predicted_labels has the same length as ground_truth_labels
    nmi_value = calculate_nmi(ground_truth_labels, predicted_labels)
    return final_communities, modularity_value, nmi_value

# Function to visualize the communities
def visualize_communities(graph, communities):
    pos = nx.spring_layout(graph)
    colors = plt.cm.rainbow(np.linspace(0, 1, len(communities)))

    for idx, community in enumerate(communities):
        nx.draw_networkx_nodes(graph, pos, nodelist=community, node_color=[colors[idx]], label=f"Community {idx+1}")
    
    nx.draw_networkx_edges(graph, pos)
    nx.draw_networkx_labels(graph, pos)
    plt.legend()
    plt.show()

# Example usage
if __name__ == '__main__':
    G = load_graph_from_mtx('karate.mtx')
    
    # Assuming ground_truth_labels is a list representing the true community labels
    ground_truth_labels = [0] * 17 + [1] * 16  # Example ground truth labels

    final_communities, modularity_value, nmi_value = lcd_sn_algorithm(G, ground_truth_labels)

    for idx, community in enumerate(final_communities):
        print(f"Community {idx + 1}: {sorted(community)}")
    
    print(f"Modularity: {modularity_value}")
    print(f"NMI: {nmi_value}")

    # Visualize the communities
    visualize_communities(G, final_communities)
