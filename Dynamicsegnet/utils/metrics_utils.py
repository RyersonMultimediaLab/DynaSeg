###Bijou
import pickle
from prettytable import PrettyTable
import math


def calculate_metrics_average(eval_results_path):
    # Load eval_results.pkl
    with open(eval_results_path, 'rb') as f:
        eval_results = pickle.load(f)

    # Calculate the average of all the metrics
    num_samples = len(eval_results)

    # Initialize variables to store the sums of each metric
    sum_aAcc = sum_mIoU = sum_mAcc = sum_aAcc_th = sum_mIoU_th = sum_mAcc_th = sum_aAcc_st = sum_mIoU_st = sum_mAcc_st = 0

    # Initialize variables to keep track of valid samples
    valid_samples = 0

    # Iterate over each sample's metrics
    for sample_metrics in eval_results.values():
        if not any(math.isnan(value) for value in sample_metrics.values()):
            valid_samples += 1
            sum_aAcc += sample_metrics['aAcc']
            sum_mIoU += sample_metrics['mIoU']
            sum_mAcc += sample_metrics['mAcc']
            sum_aAcc_th += sample_metrics['aAcc_th']
            sum_mIoU_th += sample_metrics['mIoU_th']
            sum_mAcc_th += sample_metrics['mAcc_th']
            sum_aAcc_st += sample_metrics['aAcc_st']
            sum_mIoU_st += sample_metrics['mIoU_st']
            sum_mAcc_st += sample_metrics['mAcc_st']

    # Calculate averages
    avg_aAcc = sum_aAcc / valid_samples if valid_samples > 0 else float('nan')
    avg_mIoU = sum_mIoU / valid_samples if valid_samples > 0 else float('nan')
    avg_mAcc = sum_mAcc / valid_samples if valid_samples > 0 else float('nan')
    avg_aAcc_th = sum_aAcc_th / valid_samples if valid_samples > 0 else float('nan')
    avg_mIoU_th = sum_mIoU_th / valid_samples if valid_samples > 0 else float('nan')
    avg_mAcc_th = sum_mAcc_th / valid_samples if valid_samples > 0 else float('nan')
    avg_aAcc_st = sum_aAcc_st / valid_samples if valid_samples > 0 else float('nan')
    avg_mIoU_st = sum_mIoU_st / valid_samples if valid_samples > 0 else float('nan')
    avg_mAcc_st = sum_mAcc_st / valid_samples if valid_samples > 0 else float('nan')

    # Create a PrettyTable
    table = PrettyTable()

    # Define the columns
    table.field_names = ["Category", "aAcc", "mIoU", "mAcc"]

    # Add rows for "All", "Things", and "Stuff"
    table.add_row(["All", avg_aAcc, avg_mIoU, avg_mAcc])
    table.add_row(["Things", avg_aAcc_th, avg_mIoU_th, avg_mAcc_th])
    table.add_row(["Stuff", avg_aAcc_st, avg_mIoU_st, avg_mAcc_st])

    # Print the table
    print(table)

    # Return the averages as a dictionary
    return {
        'avg_aAcc': avg_aAcc,
        'avg_mIoU': avg_mIoU,
        'avg_mAcc': avg_mAcc,
        'avg_aAcc_th': avg_aAcc_th,
        'avg_mIoU_th': avg_mIoU_th,
        'avg_mAcc_th': avg_mAcc_th,
        'avg_aAcc_st': avg_aAcc_st,
        'avg_mIoU_st': avg_mIoU_st,
        'avg_mAcc_st': avg_mAcc_st
    }
#
# def calculate_metrics_average(eval_results_path):
#     # Load eval_results.pkl
#     with open(eval_results_path, 'rb') as f:
#         eval_results = pickle.load(f)
#
#     # Calculate the average of all the metrics
#     num_samples = len(eval_results)
#
#     # Initialize variables to store the sums of each metric
#     sum_aAcc = sum_mIoU = sum_mAcc = sum_aAcc_th = sum_mIoU_th = sum_mAcc_th = sum_aAcc_st = sum_mIoU_st = sum_mAcc_st = 0
#
#     # Iterate over each sample's metrics
#     for sample_metrics in eval_results.values():
#         sum_aAcc += sample_metrics['aAcc']
#         sum_mIoU += sample_metrics['mIoU']
#         sum_mAcc += sample_metrics['mAcc']
#         sum_aAcc_th += sample_metrics['aAcc_th']
#         sum_mIoU_th += sample_metrics['mIoU_th']
#         sum_mAcc_th += sample_metrics['mAcc_th']
#         sum_aAcc_st += sample_metrics['aAcc_st']
#         sum_mIoU_st += sample_metrics['mIoU_st']
#         sum_mAcc_st += sample_metrics['mAcc_st']
#
#     # Calculate averages
#     avg_aAcc = sum_aAcc / num_samples
#     avg_mIoU = sum_mIoU / num_samples
#     avg_mAcc = sum_mAcc / num_samples
#     avg_aAcc_th = sum_aAcc_th / num_samples
#     avg_mIoU_th = sum_mIoU_th / num_samples
#     avg_mAcc_th = sum_mAcc_th / num_samples
#     avg_aAcc_st = sum_aAcc_st / num_samples
#     avg_mIoU_st = sum_mIoU_st / num_samples
#     avg_mAcc_st = sum_mAcc_st / num_samples
#
#     # Create a PrettyTable
#     table = PrettyTable()
#
#     # Define the columns
#     table.field_names = ["Category", "aAcc", "mIoU", "mAcc"]
#
#     # Add rows for "All", "Things", and "Stuff"
#     table.add_row(["All", avg_aAcc, avg_mIoU, avg_mAcc])
#     table.add_row(["Things", avg_aAcc_th, avg_mIoU_th, avg_mAcc_th])
#     table.add_row(["Stuff", avg_aAcc_st, avg_mIoU_st, avg_mAcc_st])
#
#     # Print the table
#     print(table)
#
#
#     # Return the averages as a dictionary
#     return {
#         'avg_aAcc': avg_aAcc,
#         'avg_mIoU': avg_mIoU,
#         'avg_mAcc': avg_mAcc,
#         'avg_aAcc_th': avg_aAcc_th,
#         'avg_mIoU_th': avg_mIoU_th,
#         'avg_mAcc_th': avg_mAcc_th,
#         'avg_aAcc_st': avg_aAcc_st,
#         'avg_mIoU_st': avg_mIoU_st,
#         'avg_mAcc_st': avg_mAcc_st
#     }
