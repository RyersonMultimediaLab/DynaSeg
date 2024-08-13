import numpy as np
import glob
import argparse
import tqdm
import cv2
import os.path

def calculate_mIOU():
    """
    This script calculates the mean Intersection over Union (mIOU) for segmentation masks.
    It takes as input a directory containing predicted segmentation masks and a directory
    containing corresponding ground truth masks. Optionally, it can resize the input images
    for evaluation purposes.

    The mIOU metric is computed by comparing each predicted segmentation mask with its
    corresponding ground truth mask. The average mIOU across all masks is then calculated
    and printed to the console.

    Command Line Arguments:
    --use_bsd500: Use BSD500 dataset for evaluation (default: True)
    --mode: Evaluation mode for BSD500 dataset (1: all, 2: fine, 3: coarse) (default: 1)
    --ground_truth_dir: Directory containing ground truth masks
    --results_dir: Directory containing predicted segmentation masks
    --resize_images: Resize input images for evaluation (default: False)
    """

    # Parsing command line arguments
    parser = argparse.ArgumentParser(description='Calculate mIOU for segmentation masks')
    parser.add_argument('--use_bsd500', action='store_true', default=True,
                        help='Use BSD500 dataset for evaluation (default: True)')
    parser.add_argument('--mode', type=int, default=1,
                        help='Evaluation mode for BSD500 dataset (1: all, 2: fine, 3: coarse) (default: 1)')
    parser.add_argument('--ground_truth_dir', metavar='N', default=None, type=str,
                        help='Directory containing ground truth masks')
    parser.add_argument('--results_dir', metavar='T', default=None, type=str,
                        help='Directory containing predicted segmentation masks')
    parser.add_argument('--resize_images', action='store_true', default=False,
                        help='Resize input images for evaluation (default: False)')
    args = parser.parse_args()

    # Getting list of input images
    results_list = sorted(glob.glob(args.results_dir + '/*'))
    mIOU_list = []

    # Iterating over each predicted segmentation mask
    for result_file in tqdm.tqdm(results_list):
        # Reading predicted segmentation mask
        predicted_mask = np.loadtxt(result_file, delimiter=',')

        # Loading corresponding ground truth mask
        if args.use_bsd500:  # Using BSD500 dataset
            no_background_flag = False
            gt_masks = []
            for i in range(100):
                gt_file = args.ground_truth_dir + "/" + result_file.split("/")[-1][:-4] + "-" + str(i) + ".csv"
                if not os.path.exists(gt_file):
                    break
                gt_masks.append(np.loadtxt(gt_file, delimiter=','))
            if args.mode == 2:
                gt_masks = gt_masks[np.argmax(np.array([len(np.unique(mask)) for mask in gt_masks]))]
            elif args.mode == 3:
                gt_masks = gt_masks[np.argmin(np.array([len(np.unique(mask)) for mask in gt_masks]))]
            gt_masks = np.array(gt_masks)
        else:  # For other datasets
            gt_masks = cv2.imread(args.ground_truth_dir + result_file[-16:-3] + "png", -1)

        # Resizing predicted mask if required
        if args.resize_images:
            predicted_mask = cv2.resize(predicted_mask, (gt_masks.shape[1], gt_masks.shape[0]), interpolation=cv2.INTER_NEAREST)

        # Handling different shapes of ground truth masks
        if len(gt_masks.shape) == 2:
            gt_masks = [gt_masks]

        # Iterating over each ground truth mask
        for gt_mask in gt_masks:
            label_list = np.unique(gt_mask)
            gg = np.zeros(gt_mask.shape)
            gt_mask_binary = np.where(gt_mask > 0, 1, gg)
            overlap_mask = gt_mask_binary * predicted_mask

            # Calculating mIOU for each label
            gt_mask_1d = gt_mask.reshape((gt_mask.shape[0]*gt_mask.shape[1]))
            predicted_mask_1d = predicted_mask.reshape((predicted_mask.shape[0]*predicted_mask.shape[1]))
            for label in label_list:
                if no_background_flag and (label == 0):
                    continue
                label_indices = np.where(gt_mask_1d == label)[0]
                predicted_labels = predicted_mask_1d[label_indices]
                unique_predicted_labels = np.unique(predicted_labels)
                intersection_counts = [np.sum(predicted_labels == ulabel) for ulabel in unique_predicted_labels]
                union_counts = [len(label_indices) + np.sum(predicted_mask_1d == ulabel) - np.sum(predicted_labels == ulabel) for ulabel in unique_predicted_labels]
                mious = intersection_counts / np.array(union_counts, dtype='float')
                mIOU_list.append(np.max(mious))

    # Calculating and printing average mIOU
    avg_mIOU = sum(mIOU_list) / float(len(mIOU_list))
    print("Average mIOU:", avg_mIOU)

if __name__ == "__main__":
    calculate_mIOU()
