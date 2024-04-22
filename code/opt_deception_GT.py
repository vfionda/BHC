from cdlib import algorithms
import networkx as nx
import numpy as np
import itertools as it
import multiprocess
import os
from operator import itemgetter

def computeCommunities(detection_algo,g):
    if detection_algo == "leid":
        communities = algorithms.leiden(g)
    elif detection_algo == "infomap":
        communities = algorithms.infomap(g)
    elif detection_algo == "walk":
        communities = algorithms.walktrap(g)
    elif detection_algo == "greedy":
        communities = algorithms.greedy_modularity(g)
    elif detection_algo == "paris":
        communities = algorithms.paris(g)
    elif detection_algo == "combo":
        communities = algorithms.pycombo(g)
    elif detection_algo == "eig":
        communities = algorithms.eigenvector(g)
    return communities

def getDeceptionScore(coms, target_community, g):
    number_communities=len(coms.communities)
    #number of the targetCommunity members in the various communities
    member_for_community=np.zeros(number_communities, dtype=int)
    for i in range(number_communities):
        for node in coms.communities[i]:
            if node in target_community:
                member_for_community[i]+=1
    #ratio of the targetCommunity members in the various communities
    ratio_community_members=[members_for_c/len(com) for (members_for_c,com) in zip(member_for_community,coms.communities)]
    ##In how many commmunities are the members of the target spread?
    spread_members=sum([1 if mc>0 else 0 for mc in member_for_community])
    second_part = 1 / 2 * ((spread_members - 1) / number_communities) + 1/2 * (1 - sum(ratio_community_members) / spread_members)
    #####
    num_components = nx.number_connected_components(g.subgraph(target_community)) #induced subraph sonly on target community nodes
    first_part = 1 - ((num_components - 1) / (len(target_community) - 1))
    dec_score =first_part * second_part
    return dec_score

def evalUpdates(item):
    mods=item[0]
    graph=item[1]
    target_community=item[2]
    delsI=item[3]
    addsI=item[4]
    addsE=item[5]
    delsE=item[6]
    algo=item[7]
    g=graph.copy()
    currDelsI=[]
    currDelsE=[]
    currAddsI=[]
    currAddsE=[]
    for mod in mods:
        if int(mod)<len(delsI):
            edge=delsI[mod]
            currDelsI.append(edge)
            g.remove_edge(edge[0], edge[1])
        elif int(mod)<(len(delsI)+len(addsI)):
            edge=addsI[mod-len(delsI)]
            currAddsI.append(edge)
            g.add_edge(edge[0], edge[1])
        elif int(mod)<(len(delsI)+len(addsI)+len(addsE)):
            edge=addsE[mod-len(delsI)-len(addsI)]
            currAddsE.append(edge)
            g.add_edge(edge[0], edge[1])
        else:
            edge=delsE[mod-len(delsI)-len(addsI)-len(addsE)]
            currDelsE.append(edge)
            g.add_edge(edge[0], edge[1])
    
    max_dec=0.0
    min_dec=1.0
    for i in range (3):
        communities_new=computeCommunities(algo,g)
        dec=getDeceptionScore(communities_new,target_community,g)
        if dec>max_dec:
            max_dec=dec
        if dec<min_dec:
            min_dec=dec
    return (min_dec,max_dec,currDelsI,currAddsI,currAddsE,currDelsE)

def getGroundThruthCommunities(file_path)->list[list]:
    # Initialize a dictionary to hold the community IDs as keys and list of node IDs as values
    community_dict = {}
    print(file_path)

    # Read the file and update the dictionary
    with open(file_path, 'r') as file:
        for line in file:
            # Split each line into node ID and community ID
            node_id, community_id = map(int, line.strip().split())
            # If the community ID is not in the dictionary, add it with the node ID as the first element of the list
            if community_id not in community_dict:
                community_dict[community_id] = [node_id]
            else:
                # If the community ID is already in the dictionary, append the node ID to the list
                community_dict[community_id].append(node_id)

    # Convert the dictionary values to a list of lists
    communities = list(community_dict.values())

    return communities



dataset_path="./dataset/lfr"
#detection_algorithms=["leid","infomap", "walk", "greedy", "eig", "paris","combo"]
#detection_algorithms=["greedy"]
#detection_algorithms=["leid","infomap", "walk", "greedy", "paris","combo"]
detection_algorithms=["eig","paris", "combo"]
#detection_algorithms=["eig"]


budget_updates = [1, 2, 3]

res_path="./results/GT50"

for folder_name in os.listdir(dataset_path):
    folder_path = os.path.join(dataset_path, folder_name)
    if os.path.isdir(folder_path):
        graph_file_path = None
        members_file_path = None

        # Search for the graph and members files
        for file_name in os.listdir(folder_path):
            if file_name.startswith('graph_'):
                graph_file_path = os.path.join(folder_path, file_name)
            elif file_name.startswith('membership_'):
                members_file_path = os.path.join(folder_path, file_name)
        print(graph_file_path)
        g = nx.read_edgelist(graph_file_path, nodetype=int)

        communitiesGT=getGroundThruthCommunities(members_file_path)

        print(graph_file_path[0:graph_file_path.rfind('.')], g.number_of_nodes(), g.number_of_edges())

        #os.mkdir('./results/GT50'+graph_file_path[graph_file_path.rfind('/'):graph_file_path.rfind('.')])

        for algo in detection_algorithms:
            with open('./results/GT50'+graph_file_path[graph_file_path.rfind('/'):graph_file_path.rfind('.')]+'/'+algo+'.txt', 'a') as f:
                coms=computeCommunities(algo,g)
                ncom=len(coms.communities)
                print("number of communities "+str(ncom))
                for i in range(len(communitiesGT)):
                    f.write("community "+str(i)+" "+str(communitiesGT[i])+"\n")
                f.write("-------------------"+"\n")
                f.flush()
                for i in range(len(communitiesGT)):
                    target_community=communitiesGT[i]
                    best_communities=coms
                    best_dec_score=0
                    if len(target_community)>1:
                        best_dec_score=getDeceptionScore(coms,target_community,g)
                    print("community "+str(i)+" algorithm ",algo," initial dec score "+str(best_dec_score))
                    print(target_community)
                    f.write("community "+str(i)+" "+str(target_community)+" initial dec score "+str(best_dec_score)+"\n")
                    if len(target_community) > 1:
                        delsI=[]
                        delsE=[]
                        addsI=[]
                        addsE=[]
                        bestDelsI=[]
                        bestDelsE=[]
                        bestAddsI=[]
                        bestAddsE=[]
                        for member1 in target_community:
                            for member2 in target_community:
                                if g.has_edge(member1,member2) and (member1<member2):
                                    delsI.append([member1,member2])
                                elif member1<member2:
                                    addsI.append([member1,member2])
                        for member1 in target_community:
                            for nmember in g:
                                if (nmember not in target_community) and (g.has_edge(member1,nmember)==False):
                                    addsE.append([member1,nmember])
                                elif (nmember not in target_community):
                                    delsE.append([member1,nmember])
                        totNMod=len(delsI)+len(addsI)+len(addsE)+len(delsE)
                        for budget in budget_updates:
                            try:
                                all_mods=list(it.combinations(range(totNMod), budget))
                                items=[(mods,g,target_community,delsI,addsI,addsE,delsE,algo) for mods in all_mods]

                                print("budget ",budget," all combinations ", len(all_mods)," ",len(items))
                    
                                if __name__ == '__main__':
                                    with multiprocess.Pool() as pool:
                                        result=pool.map(evalUpdates,items)
                    
                                maxDec = max(result, key = itemgetter(1))
                    
                                print("community "+str(i)+" budget "+str(budget)+" best dec score "+str(maxDec[1])+" best min dec score "+str(maxDec[0]))
                                print("\t\t bestDelsI= ",maxDec[2])
                                print("\t\t bestAddsI= ",maxDec[3])
                                print("\t\t bestAddsE= ",maxDec[4])
                                print("\t\t bestDelsE= ",maxDec[5])
                                f.write("--> budget "+str(budget)+" best dec score "+str(maxDec[1])+" best min dec score "+str(maxDec[0])+"\n")
                                f.write("\t\t bestDelsI= "+str(maxDec[2])+"\n")
                                f.write("\t\t bestAddsI= "+str(maxDec[3])+"\n")
                                f.write("\t\t bestAddsE= "+str(maxDec[4])+"\n")
                                f.write("\t\t bestDelsE= "+str(maxDec[5])+"\n")
                                f.write("\n")
                                f.flush()
                            except:
                                print("community " + str(i) + " budget " + str(budget) + " error " )
                                f.write("--> budget " + str(budget) + " error \n")

