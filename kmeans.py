import torch

'''Implements the K-Mediods algorithm for clustering'''

def pairwise_dist(A, B):
    n_1, n_2 = A.shape[0], B.shape[0]
    norms_1 = torch.sum(A.pow(2), dim=1, keepdim=True)
    norms_2 = torch.sum(B.pow(2), dim=1, keepdim=True)
    norms = norms_1.expand(n_1, n_2) + \
            norms_2.transpose(0, 1).expand(n_1, n_2)
    return torch.abs(norms - 2 * A.mm(B.t()))


def kmediods(X, n_clusters):

    X = X.float()
    centers = 10 * torch.randn(n_clusters, X.shape[1])
    last_cluster_assignment = torch.zeros(X.shape[0]).long()

    while True:

        # compute distances to centers and assign clusters
        dist = pairwise_dist(X, centers)
        cluster_assignment = torch.argmin(dist, dim=1)

        # check convergence
        if bool(torch.all(torch.eq(last_cluster_assignment, cluster_assignment))):
            break

        last_cluster_assignment = cluster_assignment

        # compute new centers
        for k in range(n_clusters):
            mean = X[cluster_assignment == k].mean(dim=0)
            mediod_index = torch.argmin((X - mean.unsqueeze(0)).pow(2).sum(1), dim=0)
            centers[k] = X[mediod_index]

    return cluster_assignment, centers













# import torch


# def pairwise_distance(data1, data2=None):
# 	r'''
# 	using broadcast mechanism to calculate pairwise ecludian distance of data
# 	the input data is N*M matrix, where M is the dimension
# 	we first expand the N*M matrix into N*1*M matrix A and 1*N*M matrix B
# 	then a simple elementwise operation of A and B will handle the pairwise operation of points represented by data
# 	'''

# 	#N*1*M
# 	A = data1.unsqueeze(dim=1)

# 	#1*N*M
# 	B = data2.unsqueeze(dim=0)

# 	dis = (A-B)**2.0
# 	#return N*N matrix for pairwise distance
# 	dis = dis.sum(dim=-1).squeeze()
# 	return dis


# def forgy(X, n_clusters):
# 	_len = len(X)
# 	indices = np.random.choice(_len, n_clusters)
# 	initial_state = X[indices]
# 	return initial_state


# def kmeans(X, n_clusters):
# 	X = X.float()

# 	centers = forgy(X, n_clusters)

# 	last_cluster_choice = torch.zeros(X.shape[0]).long()

# 	while True:

# 		dis = pairwise_distance(X, initial_state)

# 		if torch.any(torch.isnan(dis)):
# 			print('NAN dis')
# 			exit(0)

# 		choice_cluster = torch.argmin(dis, dim=1)
# 		print('dis')
# 		print(dis.shape)

# 		# Stopping conditions -- stop if nothing changes (or all change)

# 		nothing_changed = bool(
# 			torch.all(torch.eq(last_cluster_choice, choice_cluster)))

# 		# really hacky but not aware of a better way right now
# 		diff = torch.abs(choice_cluster - last_cluster_choice)
# 		changed_all = False
# 		for k in range(1, n_clusters + 1):
# 			changed_all = changed_all or bool(
# 				torch.all(torch.eq(diff, k * torch.ones(diff.shape).long())))

# 		if nothing_changed or changed_all:
# 			break

# 		last_cluster_choice = choice_cluster

# 		# Lloyds

# 		initial_state_pre = initial_state.clone()

# 		for index in range(n_clusters):
# 			selected = torch.nonzero(choice_cluster == index).squeeze()

# 			selected = torch.index_select(X, 0, selected)
# 			initial_state[index] = selected.mean(dim=0)

# 			if torch.any(torch.isnan(initial_state)):
# 				print()
# 				print('NAN')
# 				print(initial_state)
# 				print(index)
# 				print(selected.mean(dim=0))
# 				print(selected)
# 				print(choice_cluster == index)
# 				print(choice_cluster)
# 				exit(0)

# 		# center_shift = torch.sum(torch.sqrt(torch.sum((initial_state - initial_state_pre) ** 2, dim=1)))

# 	return choice_cluster, initial_state
