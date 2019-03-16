import torch
import copy
import torch.distributions as ds

'''K-Mediods algorithm for clustering'''

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


'''Hierarchical Agglomerative Clustering (HAC)'''

# min distance
def dist_min(c1, c2, f):
	min_dist = float('inf')
	for p1 in c1:
		for p2 in c2:
			new_dist = torch.norm(p1 - p2, p=f)
			if new_dist < min_dist:
				min_dist = new_dist
	return min_dist

# max distance
def dist_max(c1, c2, f):
	max_dist = - float('inf')
	for p1 in c1:
		for p2 in c2:
			new_dist = torch.norm(p1 - p2, p=f)
			if new_dist > max_dist:
				max_dist = new_dist
	return max_dist

# centroid distance
def dist_centroid(c1, c2, f):
	return torch.norm(c1.mean(0) - c2.mean(0), p=f)

# average distance
def dist_ave(c1, c2, f):
	sum = 0
	for p1 in c1:
		for p2 in c2:
			sum += torch.norm(p1 - p2, p=f)
	return sum / (len(c1) * len(c2))


'''
HAC algorithm

Inputs
- modes is shape (N, dim)
- sizes is list of cluster sizes to be returned 
- dist is one of the above distance functions
- f is float defining norm

Return
- clusters is list of lists of clusters for all sizes of input sizes
- merge_dists: list of all merging distances

'''


def hac(modes, sizes, dist, f=1.0):

	# 1 - every particle is a cluster
	stored_clusters = []
	clusters = []
	merge_dists = []
	for w in modes:
		clusters.append(w.unsqueeze(0))

	# 2 - merge closest clusters until there is only one cluster left
	for n in range(len(clusters) - 1):

		merge_i = -1
		merge_j = -1
		min_dist = torch.tensor(float('inf'))

		# 2.1 - find closest pair of cluster
		for i in range(len(clusters)):
			for j in range(i + 1, len(clusters)):
				new_dist = dist(clusters[i], clusters[j], f)
				if new_dist < min_dist:
					min_dist = new_dist
					merge_i = i
					merge_j = j

		# 2.2 - merge closest clusters
		clusters[merge_i] = torch.cat([clusters[merge_i], clusters[merge_j]], dim=0)
		del clusters[merge_j]
		merge_dists.append(min_dist.item())

		# 2.3 - store cluster
		if len(clusters) in sizes:
			stored_clusters.append(clusters.copy())
	
	return stored_clusters, merge_dists


# Find best K modes 
def hac_reps(modes, K, dist, f=1.0, verbose=True):

	clusters, merge_dists = hac(modes, [K], dist, f=f)

	if verbose:
		for j in range(0, len(modes) - 1):
			print('Clusters: {:4}      Last merge dist: {}'.format(len(modes) - j - 1, round(merge_dists[j], 3)))
		
		print('\nReturned cluster of size {} has components: {}'.format(K, [c.shape[0] for c in clusters[0]]))


	return clusters[0]




if __name__ == '__main__':

	
	modes = torch.cat([
		ds.MultivariateNormal(torch.zeros(2), 0.1 * torch.eye(2)).sample(torch.Size([10])),
		ds.MultivariateNormal(5 * torch.ones(2), 0.1 * torch.eye(2)).sample(torch.Size([6])),
		ds.MultivariateNormal(-3 * torch.ones(2), 0.1 * torch.eye(2)).sample(torch.Size([6]))
		], dim=0)

	sizes = list(range(1, 5))
	stored_clusters, merge_dists = hac(modes, sizes, dist_ave, f=0.4)

	K = 3
	clusters = hac_reps(modes, K, dist_ave, f=0.4, verbose=True)











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
