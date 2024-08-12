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

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out

class FPN(nn.Module):
    def __init__(self, num_channels):
        super(FPN, self).__init__()
        self.conv6 = nn.Conv2d(num_channels, num_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        x = self.conv6(x)
        return x


@ARCHITECTURES.register_module()
class MyCustomArchitectureOptimalClustersResNet18(nn.Module):
    def __init__(self, input_dim, T, mu, update_factor, num_layers=3, qdy=64):
        super(MyCustomArchitectureOptimalClustersResNet18, self).__init__()
        self.resnet18 = models.resnet18(pretrained=True)  # Load pre-trained ResNet-18 model
        self.input_dim = input_dim
        self.T = T
        self.mu = mu
        self.update_factor = update_factor
        self.optimal_clusters = {}
        self.qdy = qdy
        num_channels = qdy

        # Replace the fully connected layer with a convolutional layer
        self.resnet18.fc = nn.Conv2d(512, input_dim, kernel_size=1, stride=1, padding=0, bias=False)

        # Initialize the rest of the layers
        self.bn1 = nn.BatchNorm2d(input_dim)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self.make_layer(ResidualBlock, input_dim, num_channels, num_layers)
        self.layer2 = self.make_layer(ResidualBlock, num_channels, num_channels * 2, num_layers)
        self.layer3 = self.make_layer(ResidualBlock, num_channels * 2, num_channels * 4, num_layers)
        self.layer4 = self.make_layer(ResidualBlock, num_channels * 4, num_channels * 8, num_layers)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fpn = FPN(num_channels)
        self.conv3 = nn.Conv2d(num_channels, num_channels, kernel_size=1, stride=1, padding=0)
        self.bn3 = nn.BatchNorm2d(num_channels)

    def make_layer(self, block, in_channels, out_channels, num_layers, stride=1):
        layers = []
        layers.append(block(in_channels, out_channels, stride))
        for _ in range(1, num_layers):
            layers.append(block(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x1 = x
        x = self.resnet18.conv1(x)
        x = self.resnet18.bn1(x)
        x = self.resnet18.relu(x)
        x = self.resnet18.layer1(x)
        fpn_output = self.fpn(x)
        feature_map = self.conv3(fpn_output)
        normalized_feature_map = self.bn3(feature_map)
        normalized_feature_map = self.conv3(normalized_feature_map)
        normalized_feature_map = self.bn3(normalized_feature_map)
        # Adjust output size to match input size
        normalized_feature_map = self.adjust_output_size(normalized_feature_map, x1.size()[2:])

        # Argmax to obtain cluster labels
        _, labels = torch.max(normalized_feature_map, dim=1)

        return normalized_feature_map, labels
    def adjust_output_size(self, output, input_size):
        # Compare model output size with input size
        output_size = output.size()[2:]
        diff_h = output_size[0] - input_size[0]
        diff_w = output_size[1] - input_size[1]

        if diff_h > 0:  # Output height is larger than input height
            # Crop the top and bottom rows
            output = output[:, :, diff_h // 2:-diff_h // 2, :]
        elif diff_h < 0:  # Output height is smaller than input height
            # Pad top and bottom rows with zeros
            pad_top = abs(diff_h) // 2
            pad_bottom = abs(diff_h) - pad_top
            output = F.pad(output, (0, 0, pad_top, pad_bottom))

        if diff_w > 0:  # Output width is larger than input width
            # Crop the left and right columns
            output = output[:, :, :, diff_w // 2:-diff_w // 2]
        elif diff_w < 0:  # Output width is smaller than input width
            # Pad left and right columns with zeros
            pad_left = abs(diff_w) // 2
            pad_right = abs(diff_w) - pad_left
            output = F.pad(output, (pad_left, pad_right, 0, 0))

        return output

#    def forward(self, x):

        # Enable anomaly detection
    #        torch.autograd.set_detect_anomaly(True)

        # Feature extraction using the backbone (ResNet-18)
        #        r = self.backbone(x)
        #        print(f"Size after backbone: {r.size()}")

        # Apply FPN
        #        r = [r] * self.M  # Assuming M is the number of conv components
        #        fpn_outputs = self.fpn(r)

        # Final classification layer
        #        rdy = self.classifier(fpn_outputs[-1].clone())

        # Normalize the response map
        #        logits = self.bn(rdy)

        # Argmax to obtain cluster labels
        #        _, labels = torch.max(rdy, dim=1)
    #        return logits, labels

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
