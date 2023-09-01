#Generation of sysntetic networks

import networkx as nx
import os

path="./dataset/"
# Create a folder
os.mkdir(os.path.join(path,"ErdosRenyi"))
os.mkdir(os.path.join(path,"BarabasiAlbert"))

# Write a sample file to Google Drive


# Define the number of nodes and edges
num_nodes = [20,40]
num_edges = [50,100]
num_edgesBA = [2,2]

for i in range (len(num_nodes)):
  # Generate the Erdős-Rényi graphs
  graphER = nx.gnm_random_graph(num_nodes[i], num_edges[i], directed=False)
  # Create Barabasi-Albert graph
  graphBA = nx.barabasi_albert_graph(num_nodes[i], num_edgesBA[i])

  # Save the graph to a file
  nx.write_edgelist(graphER, os.path.join(path,'ErdosRenyi/undirected/Erdos-Renyi-'+str(num_nodes[i])+'-'+str(num_edges[i])+'.txt'))
  nx.write_edgelist(graphBA, os.path.join(path,'BarabasiAlbert/Barabasi-Albert-'+str(num_nodes[i])+'-'+str(num_edgesBA[i])+'.txt'))
