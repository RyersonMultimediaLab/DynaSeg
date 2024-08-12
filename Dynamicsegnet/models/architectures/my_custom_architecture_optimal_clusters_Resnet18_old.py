import torch
import torch.nn as nn
import torch.nn.functional as F
from ..builder import ARCHITECTURES
import torch.optim as optim
from mmcv.runner import Hook
from mmcv.runner import HOOKS
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import torchvision.models as models

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

class FPN(nn.Module):
    def __init__(self, in_channels_list, out_channels):
        super(FPN, self).__init__()
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(in_channels, out_channels, kernel_size=1)
            for in_channels in in_channels_list
        ])
        self.output_convs = nn.ModuleList([
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
            for _ in in_channels_list
        ])

    def forward(self, xs):
        # Compute lateral features
        lateral_features = [lateral_conv(x) for lateral_conv, x in zip(self.lateral_convs, xs)]

        # Build pyramid
        pyramid_features = [lateral_features[-1]]
        for i in range(len(lateral_features) - 1, 0, -1):
            upsampled_feature = F.interpolate(pyramid_features[-1], scale_factor=2, mode='nearest')
            pyramid_feature = lateral_features[i] + upsampled_feature
            pyramid_features.append(pyramid_feature)

        # Apply 3x3 conv to each level of the pyramid
        final_output_features = [output_conv(feature) for output_conv, feature in zip(self.output_convs, pyramid_features)]

        return final_output_features


@ARCHITECTURES.register_module()
class MyCustomArchitectureOptimalClustersResNet18(nn.Module):
    def __init__(self, M, p, qdy, T, mu, update_factor):
        super(MyCustomArchitectureOptimalClustersResNet18, self).__init__()
        self.M = M
        self.p = p
        self.qdy = qdy
        self.T = T
        self.mu = mu
        self.update_factor = update_factor
        self.optimal_clusters = {}   # Initialize optimal_clusters as an empty dictionary

        #the backbone
        # a pretrained ResNet-18
        self.backbone = models.resnet18(pretrained=True)
        in_features = self.backbone.fc.in_features
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])  # Remove the last fully connected layer

        # Add an FPN to the architecture
        in_channels_list = [512] * M  # Assuming p is the number of channels in each conv component
        self.fpn = FPN(in_channels_list, out_channels=qdy)


        # the header
        # Final classification layer
        self.classifier = nn.Conv2d(self.qdy, qdy, kernel_size=1, stride=1)

        # Batch normalization
        self.bn = nn.BatchNorm2d(qdy)

        # Define the optimizer
        self.optimizer = optim.SGD(self.parameters(), lr=0.01)

    def forward(self, x):

        # Enable anomaly detection
        torch.autograd.set_detect_anomaly(True)

        # Feature extraction using the backbone (ResNet-18)
        r = self.backbone(x)
        print(f"Size after backbone: {r.size()}")

        # Apply FPN
        r = [r] * self.M  # Assuming M is the number of conv components
        fpn_outputs = self.fpn(r)

        # Final classification layer
        rdy = self.classifier(fpn_outputs[-1].clone())

        # Normalize the response map
        logits = self.bn(rdy)

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
        loss = loss_sim + self.mu * loss_con / self.qdy  # DynaSeg SCF version
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
