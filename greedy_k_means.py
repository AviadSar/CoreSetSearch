import torch
import numpy as np
from sklearn.metrics import pairwise_distances


class GreedyKMeans:
    def __init__(self, all_pts):
        if torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')
        self.all_pts = torch.tensor(np.array(all_pts), device=device)
        self.dset_size = len(all_pts)
        self.min_distances = None
        self.already_selected = []

        # reshape
        # feature_len = self.all_pts[0].shape[1]
        # self.all_pts = self.all_pts.reshape(-1, feature_len)

        # self.first_time = True

    def update_dist(self, centers, only_new=True, reset_dist=False):
        if reset_dist:
            self.min_distances = None
        if only_new:
            centers = [p for p in centers if p not in self.already_selected]

        if centers:
            x = self.all_pts[centers]  # pick only centers
            # dist = pairwise_distances(self.all_pts, x, metric='euclidean')
            dist = ((self.all_pts - x) ** 2).sum(axis=1).reshape(-1, 1)

            if self.min_distances is None:
                self.min_distances = torch.min(dist.reshape(-1, 1), dim=1).values.reshape(-1, 1)
            else:
                self.min_distances = torch.minimum(self.min_distances, dist)

    def sample(self, already_selected, sample_size):

        # initially updating the distances
        self.update_dist(already_selected, only_new=False, reset_dist=True)
        self.already_selected = already_selected

        # epdb.set_trace()

        # pdb.set_trace()
        for i in range(sample_size):
            if not self.already_selected:
                ind = np.random.choice(np.arange(self.dset_size))
            else:
                ind = torch.argmax(self.min_distances)

            assert ind not in already_selected
            self.update_dist([ind], only_new=True, reset_dist=False)
            self.already_selected.append(ind)

            if i % 100 == 0:
                print('done {} out of {} samples'.format(i, sample_size))

        max_distance = max(self.min_distances)
        print("Max distance from cluster : %0.2f" % max_distance)

        return self.already_selected, max_distance