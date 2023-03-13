import heapq
from itertools import combinations

def dijkstra(graph, start):
    # Initialize distances and paths dictionaries with start node distance/path 0/''
    distances = {node: float('inf') for node in graph}
    distances[start] = 0
    paths = {node: '' for node in graph}
    paths[start] = start
    
    # Initialize heap with start node
    heap = [(0, start)]
    
    while heap:
        # Pop node with smallest distance from heap
        current_distance, current_node = heapq.heappop(heap)
        
        # Check if current node has already been processed
        if current_distance > distances[current_node]:
            continue
        
        # Check neighbors and update distances and paths
        for neighbor, weight in graph[current_node].items():
            distance = current_distance + weight
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                paths[neighbor] = paths[current_node] + '-' + neighbor
                heapq.heappush(heap, (distance, neighbor))
    
    return distances, paths

def std(s):
    h=s
    b=[[i.split('=')[0][0].upper(),i.split('=')[0][1].upper(),i.split('=')[-1]] for i in h]

    unic=[]
    for i in b:
        if not i[0].upper() in unic:
            unic.append(i[0].upper())
        if not i[1].upper() in unic:
            unic.append(i[1].upper())
    unic.sort()

    graph = {}
    for i in unic:
        temp={}
        for j in b:
            if j[0]==i or j[1]==i:
                if j[0]!=i:
                    temp[j[0]]=int(j[-1])
                else:
                    temp[j[1]]=int(j[-1])
        graph[i]=temp
        
    com=list(combinations(unic,2))
    final={}
    pth={}
    for i in unic:
        distances, paths = dijkstra(graph, i)
        pth[i]=paths
        del distances[i]
        final[i]=distances
    des=[]
    dis=[]
    pat=[]
    for i in com:
        des.append(i)
        dis.append(final[i[0]][i[1]])
        pat.append(pth[i[0]][i[1]])

    return des,dis,pat

