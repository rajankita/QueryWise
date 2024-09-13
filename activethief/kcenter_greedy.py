# Copyright 2017 Google Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Returns points that minimizes the maximum distance of any point to a center.

Implements the k-Center-Greedy method in
Ozan Sener and Silvio Savarese.  A Geometric Approach to Active Learning for
Convolutional Neural Networks. https://arxiv.org/abs/1708.00489 2017

Distance metric defaults to l2 distance.  Features used to calculate distance
are either raw features or if a model has transform method then uses the output
of model.transform(X).

Can be extended to a robust k centers algorithm that ignores a certain number of
outlier datapoints.  Resulting centers are solution to multiple integer program.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from sklearn.metrics import pairwise_distances
from tqdm import tqdm
import sys

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset

# from sampling_methods.sampling_def import SamplingMethod


class kCenterGreedy:

    def __init__(self, model, data, feature='fc', metric='euclidean'):

        self.model = model
        # self.labeled_loader = labeled_loader
        # self.unlabeled_loader = unlabeled_loader
        self.data = data
        # self.labeled_idx = labeled_idx
        # self.unlabeled_idx = unlabeled_idx
        self.metric = metric
        self.min_distances = None
        self.already_selected = []
        self.feature = feature        
            
        
    def update_distances(self, cluster_centers, only_new=True, reset_dist=False):
        """Update min distances given cluster centers.

        Args:
        cluster_centers: indices of cluster centers
        only_new: only calculate distance for newly selected points and update
            min_distances.
        rest_dist: whether to reset min_distances.
        """

        if reset_dist:
            self.min_distances = None
        if only_new:
            cluster_centers = [d for d in cluster_centers
                            if d not in self.already_selected]
        if cluster_centers:
            # x = self.features[cluster_centers]
            # dist = pairwise_distances(self.features, x, metric=self.metric)
            feat_labeled = self.features[cluster_centers]
            dist = pairwise_distances(self.features, feat_labeled, metric=self.metric)

            if self.min_distances is None:
                self.min_distances = np.min(dist, axis=1).reshape(-1,1)
            else:
                self.min_distances = np.minimum(self.min_distances, dist)
                
                
    def select_batch(self, labeled_idx, unlabeled_idx, N, **kwargs):
        """
        Diversity promoting active learning method that greedily forms a batch
        to minimize the maximum distance to a cluster center among all unlabeled
        datapoints.

        Args:
        model: model with scikit-like API with decision_function implemented
        already_selected: index of datapoints already selected (labeled data)
        N: batch size

        Returns:
        indices of points selected to minimize distance to cluster centers
        """
        
        # Compute features for all data
        model = self.model
        model.eval()
        
        # register forward hooks on the layers of choice
        if self.feature == 'avgpool':
            # a dict to store the activations
            activation = {}
            def getActivation(name):
                # the hook signature
                def hook(model, input, output):
                    activation[name] = output.detach()
                return hook

            h1 = model.avgpool.register_forward_hook(getActivation('avgpool'))   
        
            # compute activations for both labeled and unlabeled data
            all_idx = labeled_idx + unlabeled_idx
            all_loader = DataLoader(Subset(self.data, all_idx), batch_size=128, 
                                    pin_memory=False, num_workers=4, shuffle=True)
            feat = []
            already_selected = []  # local indices for labeled data points relative to full feature vector
            with torch.no_grad():
                ctr = 0
                for data in tqdm(all_loader):
                    inputs = data[0].cuda()
                    ind = data[2]
                    out = model(inputs)
                    # collect the activations in the correct list
                    feat.extend(activation['avgpool'].cpu().numpy())
                    for j in ind:
                        if j in labeled_idx:
                            already_selected.append(ctr)
                        ctr+=1
                        
            # detach the hooks
            h1.remove()
            self.features = np.asarray(feat)[:, :, 0, 0]
            
        elif self.feature == 'fc':        
            # compute activations for both labeled and unlabeled data
            all_idx = labeled_idx + unlabeled_idx
            all_loader = DataLoader(Subset(self.data, all_idx), batch_size=1024, 
                                    pin_memory=True, num_workers=8, shuffle=False)
            feat = []
            already_selected = []  # local indices for labeled data points relative to full feature vector
            with torch.no_grad():
                ctr = 0
                for data in tqdm(all_loader):
                    inputs = data[0].cuda()
                    ind = data[2].numpy()
                    with torch.cuda.amp.autocast():
                        out = model(inputs) 
                        # collect the activations in the correct list
                        feat.extend(out.cpu().numpy())
                    for j in ind:
                        if j in set(labeled_idx):
                            already_selected.append(ctr)
                        ctr+=1                    
            self.features = np.asarray(feat)
            
        else:
            raise(AssertionError)
        
        # Compute distances from unlabeled points to their nearest cluster centers
        self.update_distances(already_selected, only_new=False, reset_dist=True)

        # Start greedy selection of N unlabeled data points
        new_batch = []

        for _ in range(N):
            ind = np.argmax(self.min_distances)
            # true index of the selected data point in the original dataset
            true_ind = all_idx[ind]
            
            # New examples should not be in already selected since those points
            # should have min_distance of zero to a cluster center.
            assert ind not in already_selected

            self.update_distances([ind], only_new=True, reset_dist=False)
            new_batch.append(true_ind)
        print('Maximum distance from cluster centers is %0.2f' % max(self.min_distances))

        self.already_selected = already_selected

        return new_batch