#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
from queue import PriorityQueue 


# In[2]:


# Creating problem variables


# In[3]:


# Creates 10 random 3D points
points = np.random.rand(10, 2)


# In[4]:


# Random quotas, and makes their sum equal to `quota_sm`
quotas = np.random.rand(10,1)
sm = 0
for i in range(10):
    sm = sm + quotas[i]
    
quota_sm = 10
    
for i in range(10):
    quotas[i] = (quotas[i]*quota_sm)/sm  


# In[5]:


min_quota = 8


# In[6]:


# Adjacency List containing destinations of passengers
passengers = [[1,2], [2,3,4], [1,3,4], [5,6,7], [8,9], [2,1,4,9], [2,5,7], [1,2,3], [9,7,6], [1,2,3,4]]


# In[ ]:





# In[7]:


# Common Helper functions


# In[8]:


def distance(point1, point2):
    """
    Calculates the euclidean distance between two points
    """
    return np.sqrt(np.sum((point1 - point2)**2))


# In[9]:


def plot(points, best_path):
    """
    Visuals for easier understanding, plots the points and the best path.
    """
    B = len(best_path)
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)
    ax.scatter(points[:,0], points[:,1], c='r', marker='o')
    
    # Start Point
    p1x, p1y = points[best_path[0]][0], points[best_path[0]][1]
    
    ax.plot([p1x], [p1y], c='g', marker='*', ms = 20)
    
    for i in range(B-1):
        p1x, p1y = points[best_path[i]][0], points[best_path[i]][1]
        p2x, p2y = points[best_path[i+1]][0], points[best_path[i+1]][1]
        ax.arrow(p1x,p1y,p2x-p1x,p2y-p1y, length_includes_head = True, head_width=0.02, head_length=0.02, color='green')

    p1x, p1y = points[best_path[0]][0], points[best_path[0]][1]
    p2x, p2y = points[best_path[-1]][0], points[best_path[-1]][1]
    ax.arrow(p2x,p2y,p1x-p2x,p1y-p2y, length_includes_head = True, head_width=0.02, head_length=0.02, color='green')
    
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    plt.show()


# In[10]:


# Without Passengers


# In[11]:


def probability(points, unvisited, phermones, current_p, alpha, beta):
    """
    Calculates the probablities of ants to choose the next vertex.
    """
    sm = 0
    
    probabilities = np.zeros(len(unvisited))
    for i in range(len(unvisited)):
        probabilities[i] =  (phermones[current_p, unvisited[i]]**alpha)/(distance(points[current_p], points[unvisited[i]])**beta)
        sm = sm + probabilities[i]
        
    probabilities /= sm
        
    return probabilities


# In[12]:


def ACO(points, quotas, min_quota, source, ants, iterations, alpha, beta, eva_rate, Q):
    """
    Implements the Ant Colont Implementation of Quota Travelling Saleman Problem(Without Passengers)
    """
    
    # Initializing variables
    N = len(points)
    phermones = np.ones((N,N))
    best_path = None
    best_length = np.inf
    best_quota = 0
    
    # Number of iterations
    for iterate in range(iterations):
        curr_paths = []
        curr_lengths = []
        curr_quotas = []
        
        # Find a Path for each ant
        for ant in range(ants):
            visited = [False]*N
            # Every ant starts from the same source
            current_p = source
            visited[current_p] = True
            curr_path = [current_p]
            curr_length = 0
            curr_quota = quotas[source].copy()
            
            # Keeps going till the quota is met
            while curr_quota<min_quota:
                # The vertices the ant can visit next
                unvisited = []
                for i in range(0,len(visited)):
                    if not visited[i]:
                        unvisited.append(i)
                        
                # probabilites to choose the next vertex
                probabilities = probability(points, unvisited, phermones, current_p, alpha, beta)
                
                # Chooses based on the probability
                next_p = np.random.choice(unvisited, p=probabilities)
                
                curr_path.append(next_p)
                curr_length = curr_length + distance(points[current_p], points[next_p])
                curr_quota += quotas[next_p]
                
                visited[next_p] = True
                current_p = next_p
                

            curr_paths.append(curr_path)
            curr_lengths.append(curr_length)
            curr_quotas.append(curr_quota)
            
            #Local pheromone Update
            for j in range(len(curr_path)-1):
                phermones[curr_path[j],curr_path[j+1]] += (Q/curr_length)
            phermones[curr_path[-1],curr_path[0]] += (Q/curr_length)
            
            # Updates the Result
            if curr_length<best_length:
                best_length = curr_length
                best_path = curr_path
                best_quota = curr_quota
                
        # Global pheromone update
        phermones *= eva_rate
            
        # Local pheromone step can be done globally if required
#         for i in range(0,len(curr_paths)):
#             for j in range(len(curr_paths[i])-1):
#                 phermones[curr_paths[i][j],curr_paths[i][j+1]] += (Q/curr_lengths[i])
#             phermones[curr_paths[i][-1],curr_paths[i][0]] += (Q/curr_lengths[i])
                
    
    plot(points, best_path)
    
    print("Path:" + str(best_path))
    print("Quota:" + str(best_quota))
    print("Cost:" + str(best_length))
    print("Reward:0")
    print("Total Cost:" + str(best_length))


# In[13]:


ACO(points, quotas, min_quota, 0, 10, 100, 1, 1, 0.5, 1)


# In[ ]:





# In[ ]:





# In[14]:


def probability(points, passengers, unvisited, phermones, current_p, quotas, alpha, beta, gamma, delta):
    """
    Calculates the probablities of ants to choose the next vertex.
    """
    sm = 0
    
    probabilities = np.zeros(len(unvisited))
    for i in range(len(unvisited)):
        probabilities[i] =  (phermones[current_p, unvisited[i]]**alpha)*(quotas[i]**gamma)*((len(passengers[i])+1)**delta)
        probabilities[i] /= distance(points[current_p], points[unvisited[i]])**beta
        sm = sm + probabilities[i]

        
    probabilities /= sm
        
    return probabilities


# In[ ]:





# In[15]:


# Unlimited Passengers


# In[16]:


def total_cost_with_passengers(points, curr_path, passengers):
    """ For Unlimited seats, we will satisfy all the passengers"""
    freq = {}
    N = len(curr_path)
    Total_cost = 0
    Total_Reward = 0
    Dest_array = [0]*N
    num_passengers = 0
    
    for i in range(1, N):
        freq[curr_path[i]] = i
        
    for i in passengers[curr_path[0]]:
        if i in freq.keys():
            num_passengers += 1
            Dest_array[freq[i]] -= 1

    for i in range(1, N):
        curr_cost = distance(points[curr_path[i-1]], points[curr_path[i]])
        Total_cost += curr_cost
        Total_Reward += (num_passengers*curr_cost)/(num_passengers+1)
        
        del freq[curr_path[i]]
        for j in passengers[curr_path[i]]:
            if j in freq.keys():
                num_passengers += 1
                Dest_array[freq[j]] -= 1
                
        num_passengers -= Dest_array[i]
                
    return Total_cost, Total_Reward
    


# In[17]:


def ACO(points, passengers, quotas, min_quota, source, ants, iterations, alpha, beta, gamma, delta, eva_rate, Q):
    """
    Implements the Ant Colont Implementation of Quota Travelling Saleman Problem With Unlimited Passengers
    """
    
    # Initializing variables
    N = len(points)
    phermones = np.ones((N,N))
    best_path = None
    best_cost = np.inf
    best_quota = 0
    best_reward = 0
    
    # Number of Iterations
    for iterate in range(iterations):
        curr_paths = []
        curr_lengths = []
        curr_quotas = []
        
        # Find a Path for each ant
        for ant in range(ants):
            visited = [False]*N
            # Every ant starts from the same source
            current_p = source
            visited[current_p] = True
            curr_path = [current_p]
            curr_length = 0
            curr_quota = quotas[source].copy()
            
            # Keeps going till the quota is met
            while curr_quota<min_quota:
                # The vertices the ant can visit next
                unvisited = []
                for i in range(0,len(visited)):
                    if not visited[i]:
                        unvisited.append(i)
                        
                # probabilites to choose the next vertex
                probabilities = probability(points, passengers, unvisited, phermones, current_p, quotas, alpha, beta, gamma, delta)
                
                # Chooses based on the probability
                next_p = np.random.choice(unvisited, p=probabilities)
                
                curr_path.append(next_p)
                curr_length = curr_length + distance(points[current_p], points[next_p])
                curr_quota += quotas[next_p]
                
                visited[next_p] = True
                current_p = next_p
                

            curr_paths.append(curr_path)
            curr_lengths.append(curr_length)
            curr_quotas.append(curr_quota)
            
            #Local pheromone Update
            for j in range(len(curr_path)-1):
                phermones[curr_path[j],curr_path[j+1]] += (Q/curr_length)
            phermones[curr_path[-1],curr_path[0]] += (Q/curr_length)
            
            # Updates the Result
            curr_cost, curr_reward = total_cost_with_passengers(points, curr_path, passengers)
            if (curr_cost-curr_reward)<(best_cost-best_reward):
                best_cost = curr_cost
                best_path = curr_path
                best_quota = curr_quota
                best_reward = curr_reward
                
        # Global pheromone update
        phermones *= eva_rate
                
    
    plot(points, best_path)
    
    print("Path:" + str(best_path))
    print("Quota:" + str(best_quota))
    print("Cost:" + str(best_cost))
    print("Reward:" + str(best_reward))
    print("Total Cost:" + str(best_cost - best_reward))


# In[18]:


ACO(points, passengers, quotas, min_quota, 0, 10, 100, 1, 1, 1, 1, 0.5, 1)


# In[ ]:





# In[ ]:





# In[19]:


# Limited Passengers


# In[20]:


def passengers_in_a_path(start_p, end_p, cnt_mat, passenger_array, max_passengers, mx_limit=99999999):
    """
    Finds out the passengers that can travel from `start_p` to `end_p` directly or indirectly.
    For example:
    directly = `1->5`
    indirectly = `1->2->4->5` (First passenger get off on 2, second gets in which gets off on 4 and so on.)
    """
    curr = 0
    mx = 0
    for i in range(0, end_p+1):
        curr += passenger_array[i]
        if i>=start_p:
            mx = max(mx, curr)
            
    # Max number of passenger that can travel
    limit_passengers = min(max_passengers - mx, mx_limit)
    
    if limit_passengers<=0:
        return passenger_array, 0, cnt_mat
    
    # Directly
    num_passengers = cnt_mat[start_p][end_p]
    if num_passengers>=limit_passengers:
        cnt_mat[start_p][end_p] -= limit_passengers
        
        passenger_array[start_p] += limit_passengers
        if (end_p+1)<len(passenger_array):
            passenger_array[end_p] -= limit_passengers
            
        return passenger_array, limit_passengers, cnt_mat

    cnt_mat[start_p][end_p] = 0
    
    
    # Indirectly
    for i in range(start_p+1, end_p):
        start_to_i = cnt_mat[start_p][i]
        Mx_lim = min(limit_passengers - num_passengers, start_to_i)
        
        # Recursive call for finding all indirect paths
        passenger_array, i_to_end, cnt_mat = passengers_in_a_path(i, end_p, cnt_mat, passenger_array, max_passengers, Mx_lim)
        
        temp = i_to_end
        
        if (num_passengers+temp)>=limit_passengers:
            cnt_mat[start_p][i] -= (num_passengers + temp - limit_passengers)
            
            passenger_array[start_p] += limit_passengers
            return passenger_array, limit_passengers, cnt_mat
        
        cnt_mat[start_p][i] -= temp
        num_passengers += temp
        
    passenger_array[start_p] += num_passengers
        
    return passenger_array, num_passengers, cnt_mat


# In[21]:


def total_cost_with_passengers(points, curr_path, passengers, max_passengers):
    """
    Calculates the total cost in a given path given a list of passenger and the number of seats available
    """
    N = len(curr_path)
    ind_map = [0]*N
    rev_ind_map = [-1]*len(points)
    
    # Mapping for finding the correct indices
    for i in range(0,N):
        ind_map[i] = curr_path[i]
        rev_ind_map[curr_path[i]] = i
    
    cnt_mat = [[0]*N]*N
    
    # Matrix which stores the total number of passengers from point `i` to point `j` (directly)
    for i in range(0,N):
        for j in passengers[curr_path[i]]:
            if rev_ind_map[j]>i:
                cnt_mat[i][rev_ind_map[j]] += 1
                
    q = PriorityQueue()
    
    Total_cost = 0
    
    # Priority queue stores the total number of direct paths a passenger can take (sorted by distance)
    for i in range(0,N):
        temp = 0
        for j in range(i+1,N):
            temp += distance(points[curr_path[j-1]], points[curr_path[j]])
            q.put((temp, (i, j)))

        if i==0:
            Total_cost = temp
            
    Total_Reward = 0
    
    passenger_array = [0]*N
    
    Number_Of_Passengers = [0]*N
    
    # The passengers which travel more distance (hence reduce more cost) are taken first
    while not q.empty():
        a, (p1, p2) = q.get()
        passenger_array, num_passengers, cnt_mat = passengers_in_a_path(p1, p2, cnt_mat, passenger_array, max_passengers)
        Number_Of_Passengers[p1] += num_passengers
        if (p2+1)<N:
            Number_Of_Passengers[p2] -= num_passengers
            
    
    curr_passengers = Number_Of_Passengers[0]
    for i in range(1, N):
        curr_cost = distance(points[curr_path[i-1]], points[curr_path[i]])
        Total_Reward += (curr_passengers*curr_cost)/(curr_passengers+1)
        curr_passengers += Number_Of_Passengers[i]

    
    return Total_cost, Total_Reward
        


# In[22]:


def ACO(points, passengers, quotas, min_quota, source, ants, iterations, alpha, beta, gamma, delta, eva_rate, Q, max_passengers):
    """
    Implements the Ant Colont Implementation of Quota Travelling Saleman Problem(Without Passengers)
    """
    
    # Initializing Variables
    N = len(points)
    phermones = np.ones((N,N))
    best_path = None
    best_cost = np.inf
    best_quota = 0
    best_reward = 0
    
    # Number of Iterations
    for iterate in range(iterations):
        curr_paths = []
        curr_lengths = []
        curr_quotas = []
        
        # Find path for each ant
        for ant in range(ants):
            visited = [False]*N
            # Every ant starts from the same source
            current_p = source
            visited[current_p] = True
            curr_path = [current_p]
            curr_length = 0
            curr_quota = quotas[source].copy()
            
            # Keeps going till the quota is met
            while curr_quota<min_quota:
                # The vertices the ant can visit next
                unvisited = []
                for i in range(0,len(visited)):
                    if not visited[i]:
                        unvisited.append(i)
                        
                # probabilites to choose the next vertex
                probabilities = probability(points, passengers, unvisited, phermones, current_p, quotas, alpha, beta, gamma, delta)
                
                # Chooses based on the probability
                next_p = np.random.choice(unvisited, p=probabilities)
                
                curr_path.append(next_p)
                curr_length = curr_length + distance(points[current_p], points[next_p])
                curr_quota += quotas[next_p]
                
                visited[next_p] = True
                current_p = next_p
                

            curr_paths.append(curr_path)
            curr_lengths.append(curr_length)
            curr_quotas.append(curr_quota)
            
            #Local pheromone Update
            for j in range(len(curr_path)-1):
                phermones[curr_path[j],curr_path[j+1]] += (Q/curr_length)
            phermones[curr_path[-1],curr_path[0]] += (Q/curr_length)
            
            # Updates the Result
            curr_cost, curr_reward = total_cost_with_passengers(points, curr_path, passengers, max_passengers)
            if (curr_cost-curr_reward)<(best_cost-best_reward):
                best_cost = curr_cost
                best_path = curr_path
                best_quota = curr_quota
                best_reward = curr_reward
                
        # Global pheromone update
        phermones *= eva_rate
                
    
    plot(points, best_path)
    
    print("Path:" + str(best_path))
    print("Quota:" + str(best_quota))
    print("Cost:" + str(best_cost))
    print("Reward:" + str(best_reward))
    print("Total Cost:" + str(best_cost - best_reward))


# In[23]:


max_passengers = 3


# In[24]:


ACO(points, passengers, quotas, min_quota, 0, 10, 100, 1, 1, 1, 1, 0.5, 1, max_passengers)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




