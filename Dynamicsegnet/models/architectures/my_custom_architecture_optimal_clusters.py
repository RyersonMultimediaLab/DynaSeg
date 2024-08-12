import torch
import torch.nn as nn
import torch.nn.functional as F
from ..builder import ARCHITECTURES
import torch.optim as optim
from mmcv.runner import Hook
from mmcv.runner import HOOKS
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

@HOOKS.register_module()
class LossWeightStepUpdateHook(Hook):
    def __init__(self,
                 by_epoch=True,
                 interval=1,
                 steps=[9, 10],
                 gammas=[0, 1.0],
                 key_names=['loss_kernel_cross_weight']):
        self.steps = steps
        self.gammas = gammas
        self.by_epoch = by_epoch
        self.key_names = key_names
        self.interval = interval

    def before_run(self, runner):
        runner.log_buffer.output['loss_kernel_cross_weight'] = self.gammas[0]

    def before_train_epoch(self, runner):
        if not self.by_epoch or not self.every_n_epochs(runner, self.interval):
            return

        for i, step in enumerate(self.steps):
            if runner.epoch >= step and i < len(self.gammas) and i < len(self.key_names):
                runner.log_buffer.output[self.key_names[i]] = self.gammas[i]

@ARCHITECTURES.register_module()
class MyCustomArchitectureOptimalClusters(nn.Module):
    def __init__(self, M, p, qdy, T, mu, update_factor):
        super(MyCustomArchitectureOptimalClusters, self).__init__()
        self.M = M
        self.p = p
        self.qdy = qdy
        self.T = T
        self.mu = mu
        self.update_factor = update_factor
        self.optimal_clusters = {}   # Initialize optimal_clusters as an empty dictionary

        #the backbone
        # Define the convolutional components
        self.conv_components = nn.ModuleList()
        for i in range(M):
            in_channels = 3 if i == 0 else self.conv_components[i - 1][0].out_channels
            # out_channels = p//(M-i)
            out_channels = p
            conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
            relu = nn.ReLU(inplace=True)
            bn = nn.BatchNorm2d(out_channels)
            self.conv_components.append(nn.Sequential(conv, relu, bn))
        # the header
        # Final classification layer
        self.classifier = nn.Conv2d(p, qdy, kernel_size=1, stride=1)

        # Batch normalization
        self.bn = nn.BatchNorm2d(qdy)

        # Define the optimizer
        self.optimizer = optim.SGD(self.parameters(), lr=0.01)

    def forward(self, x):

        # Enable anomaly detection
        torch.autograd.set_detect_anomaly(True)

        # CNN subnetwork
        r = x
        for conv_component in self.conv_components:
            r = conv_component(r)
        #print(f"Size after backbone: {r.size()}")

        # Iterative forward-backward process
        # logits = self.classifier(rdy)
        rdy = self.classifier(r.clone())
        #print(f"Size after classifier: {rdy.size()}")

        # Normalize the response map
        logits = self.bn(rdy)
        #print(f"Size after Normalize the response map: {logits.size()}")

        # Argmax to obtain cluster labels
        _, labels = torch.max(rdy, dim=1)
        return logits, labels

    def loss_function(self, rdy, labels):
        # Feature similarity loss
        loss_sim = F.cross_entropy(rdy, labels)

        # applying the detach() method to the rdy tensor before performing
        # the spatial continuity loss calculation. This will create a new
        # tensor that is detached from the computation graph and won't
        # affect the gradient computation.
        # Spatial continuity loss
        rdy_detached = rdy.detach()
        diff_h = torch.abs(rdy_detached[:, :, 1:, :] - rdy_detached[:, :, :-1, :])
        diff_v = torch.abs(rdy_detached[:, :, :, 1:] - rdy_detached[:, :, :, :-1])
        loss_con = torch.mean(diff_h) + torch.mean(diff_v)

        # Total loss
        loss = loss_sim + self.mu * loss_con / self.qdy   # DynaSeg SCF version
        # loss = loss_sim + self.qdy * loss_con / self.mu  # DynaSeg FSF version

        return loss

    #

    def train_step(self, data_batch, optimizer, **kwargs):
        inputs = data_batch['img']
        image_names = data_batch['idx']
        optimal_clusters = data_batch['optimal_clusters'][image_names.item()]
        # Extract the integer value from the tensor
        optimal_clusters_value = int(optimal_clusters.item())
        print(f"optimal_clusters for image {int(image_names.item())}: {optimal_clusters_value}")

        optimizer.zero_grad()

        losses = []
        for image_idx in range(len(inputs)):
            image = inputs[image_idx].unsqueeze(0)
            image_name = image_names[image_idx]

            for _ in range(self.T):
                logits, labels = self.forward(image)
                loss = self.loss_function(logits, labels)
                loss.backward(retain_graph=True)
                optimizer.step()
                optimizer.zero_grad()

            unique_labels = torch.unique(labels)
            self.qdy = max(len(unique_labels), optimal_clusters_value)
            losses.append(loss.item())
            print(f"Loss for image {image_name}: {loss.item()}", f"Number of clusters is =", self.qdy)

        average_loss = sum(losses) / len(losses)
        log_vars = {'loss': average_loss}

        outputs = {
            'loss': average_loss,
            'log_vars': log_vars,
            'num_samples': len(inputs)
        }

        return outputs

    def predict_optimal_clusters(self, image, image_name):
        # Convert the image tensor to a numpy array if necessary
        image_np = image.cpu().numpy()

        # Flatten the image to a 1D array (assuming it's a 2D image)
        flattened_image = image_np.reshape(-1, image_np.shape[-1])

        if image_name not in self.optimal_clusters:
            # Calculate optimal clusters if it's not in the dictionary
            # List to store silhouette scores for different cluster counts
            silhouette_scores = []
            # Try different cluster counts and compute silhouette score for each
            for num_clusters in range(3, 27):  # You can adjust the range as needed
                kmeans = KMeans(n_clusters=num_clusters, n_init=10, random_state=0)
                cluster_labels = kmeans.fit_predict(flattened_image)
                silhouette_avg = silhouette_score(flattened_image, cluster_labels)
                silhouette_scores.append(silhouette_avg)

            # Find the cluster count with the highest silhouette score
            optimal_clusters = silhouette_scores.index(
                max(silhouette_scores)) + 3  # Add 3 because we started from 3 clusters

            # Store the optimal cluster count in the dictionary
            self.optimal_clusters[image_name] = optimal_clusters

        return optimal_clusters
