import time

import mmcv
import torch
from Dynamicsegnet.utils.comm import get_rank
from mmcv.runner import HOOKS, Hook, get_dist_info
from torch.utils.data import Dataset


import pickle
import os


@HOOKS.register_module()
class ValidateHook(Hook):
    """Validation hook.
    Args:
        dataset (Dataset | dict): A PyTorch dataset or dict that indicates
            the dataset.
        dist_mode (bool): Use distributed evaluation or not. Default: True.
        initial (bool): Whether to evaluate before the training starts.
            Default: True.
        interval (int): Evaluation interval (by epochs). Default: 1.
        **eval_kwargs: Evaluation arguments fed into the evaluate function of
            the dataset.
    """

    def __init__(self, initial=True, interval=1, trial=-1):
        self.initial = initial
        self.interval = interval
        self.trial = trial

    def before_run(self, runner):
        if self.initial:
            self._run_validate(runner)

    def after_train_epoch(self, runner):
        if not self.every_n_epochs(runner, self.interval):
            return
        self._run_validate(runner)

    @torch.no_grad()
    def _run_validate(self, runner):
        runner.model.eval()
        runner.evaluator.reset()
        dataloader = runner.val_dataloader
        rank, world_size = get_dist_info()
        if rank == 0:
            prog_bar = mmcv.ProgressBar(len(dataloader.dataset))

        time.sleep(2)  # This line can prevent deadlock problem in some cases.
        # for i, data in enumerate(dataloader):
        #     outputs = runner.model(mode='test', **data)
        #     outputs = runner.model(data['img'])
        for i, data in enumerate(dataloader):
            try:
                outputs = runner.model(mode='test', **data)
            except Exception:
                outputs = runner.model(data['img'])[0]

            runner.evaluator.process(data, outputs)
            if rank == 0:
                batch_size = len(outputs) * world_size
                for _ in range(batch_size):
                    prog_bar.update()
            if self.trial != -1 and i > self.trial:
                break
        ####Bijou
        # Create the output directory if it doesn't exist
        output_dir = '/home/bijouub/PycharmProjects/DynamicSegNet/output_eval'
        os.makedirs(output_dir, exist_ok=True)

        with open(
                '/home/bijouub/PycharmProjects/DynamicSegNet/tools/data/curated/val2017/Coco164kFull_Stuff_Coarse_7.txt',
                'r') as f:
            name = f.read().strip()

        eval_results = runner.evaluator.evaluate()
        print("Keys in eval_results:", eval_results.keys())

        if runner.epoch == 0:

            # Save eval_results using pickle with the filename
            with open(os.path.join(output_dir, f'{name}_eval_results.pkl'), 'wb') as f:
                pickle.dump(eval_results, f)
        else:
            # Define a placeholder for existing_eval_results
            existing_eval_results = None

            # Load the existing pickle with the filename if it exists
            eval_results_path = os.path.join(output_dir, f'{name}_eval_results.pkl')
            if os.path.exists(eval_results_path):
                with open(eval_results_path, 'rb') as f:
                    existing_eval_results = pickle.load(f)

            # Compare mIoU values
            if existing_eval_results is None or eval_results['mIoU'] > existing_eval_results['mIoU']:
                # Replace the contents of the file with the new eval_results
                existing_eval_results = eval_results
                with open(eval_results_path, 'wb') as f:
                    pickle.dump(existing_eval_results , f)

            if (runner.epoch +1) == runner.max_epochs:
                # Load the best_eval_results eval_results.pkl file if it exists
                best_eval_results = {}
                eval_results_path = '/home/bijouub/PycharmProjects/DynamicSegNet/output_eval/eval_results.pkl'

                # Check if the file exists and is not empty
                if os.path.exists(eval_results_path) and os.path.getsize(eval_results_path) > 0:
                    with open(eval_results_path, 'rb') as f:
                        best_eval_results = pickle.load(f)


                # Add the current eval_results to the best_eval_results
                best_eval_results[name] = existing_eval_results

                # Save the updated eval_results.pkl file
                with open(eval_results_path, 'wb') as f:
                    pickle.dump(best_eval_results, f)








        # best_mIoU = -1  # Initialize the best mIoU variable
        #
        # eval_results = runner.evaluator.evaluate()
        # current_mIoU = eval_results['mIoU']
        # if runner.epoch == 0 or current_mIoU > best_mIoU:
        #     best_mIoU = current_mIoU  # Update the best mIoU
        #     best_eval_results = eval_results
        #
        #     # Read the name from the file
        #     with open(
        #             '/home/boujub/PycharmProjects/DynamicSegNet/tools/data/curated/val2017/Coco164kFull_Stuff_Coarse_7.txt',
        #             'r') as f:
        #         name = f.read().strip()
        #
        #     # Create the output directory if it doesn't exist
        #     output_dir = '/home/boujub/PycharmProjects/DynamicSegNet/output_eval'
        #     os.makedirs(output_dir, exist_ok=True)
        #
        #     # Save eval_results using pickle with the filename
        #     with open(os.path.join(output_dir, f'{name}_eval_results.pkl'), 'wb') as f:
        #         pickle.dump(eval_results, f)

        for name, val in eval_results.items():
            runner.log_buffer.output[name] = val
        runner.log_buffer.ready = True
        runner.model.train()
