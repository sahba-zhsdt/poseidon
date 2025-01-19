import argparse
import torch
import numpy as np
import random
import psutil
import os
import pandas as pd
import wandb
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
from transformers.trainer_utils import EvalPrediction
from scOT.model import ScOT
from scOT.trainer import TrainingArguments, Trainer
from scOT.problems.Mybase import get_dataset, BaseTimeDataset
from scOT.metrics import relative_lp_error, lp_error
from scOT.parsers import read_parser
SEED = 0
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

def create_predictions_plot_infer(inputs, predictions, labels, **kwargs):
    # assert predictions.shape[0] >= 4

    indices = random.sample(range(predictions.shape[0]), 1)
    predictions = predictions[indices]
    labels = labels[indices]
    inputs = inputs[None, :,:,:]
    
    i_time = kwargs['initial_time']
    o_time = kwargs['final_time']
    dt = o_time - i_time

    fig = plt.figure(figsize=(40, 35))
    grid = ImageGrid(
        fig, 111, nrows_ncols=(predictions.shape[1], 3), axes_pad=0.5, 
        cbar_mode="each", cbar_pad=0.2, cbar_size="4%", cbar_location="right",
        share_all=True
    )
    
    # for-loop to get the correct [vmax, vmin] for each channel
    vmax=[]
    vmin=[]
    for i in range(predictions.shape[1]):
        vmax.append(max(predictions[:,i,:,:].max(), labels[:,i,:,:].max(), inputs[:,i,:,:].max()))
        vmin.append(min(predictions[:,i,:,:].min(), labels[:,i,:,:].min(), inputs[:,i,:,:].min()))
    
    
    for _i, ax in enumerate(grid):
        i = _i // 1 #num of sample
        j = _i % 1  #num of sample

        if _i % 3 == 0:
            im = ax.imshow(
                inputs[j, i // 3].T,
                cmap="gist_ncar",
                origin="lower",
                vmin=vmin[i//3],
                vmax=vmax[i//3],
            )
            ax.set_title(f"Initial State channel {i//3} @ time step {i_time}")
            cbar = grid.cbar_axes[_i].colorbar(im)
            
        elif _i % 3 == 1:
            im = ax.imshow(
                labels[j, i // 3].T,
                cmap="gist_ncar",
                origin="lower",
                vmin=vmin[i//3],
                vmax=vmax[i//3],
            )
            ax.set_title(f"Ground Truth channel {i//3} @ time step {o_time}")
            cbar = grid.cbar_axes[_i].colorbar(im)
            
        else:
            im = ax.imshow(
                predictions[j, i // 3].T,
                cmap="gist_ncar",
                origin="lower",
                vmin=vmin[i//3],
                vmax=vmax[i//3],
            )
            ax.set_title(f"Predicted channel {i//3} @ time step {o_time}")
            cbar = grid.cbar_axes[_i].colorbar(im)

        ax.set_xticks([])
        ax.set_yticks([])
    
    plt.suptitle(f"Prediction after {dt} time steps", fontsize=25)
    plt.savefig("./predictionB.png")

def get_test_set(
    dataset, data_path, initial_time=None, final_time=None, dataset_kwargs={}
):
    """
    Get a test set (input at initial_time, output at final_time).

    Args:
        dataset: str
            Dataset name.
        data_path: str
            Path to data.
        initial_time: int
            Initial time step to start from.
        final_time: int
            Final time step to end at.
        dataset_kwargs: dict
            Additional arguments for dataset as in scOT.problems.base.get_dataset.
    """
    if initial_time is not None and final_time is not None:
        dataset_kwargs = {
            **dataset_kwargs,
            "fix_input_to_time_step": initial_time,
            "time_step_size": final_time - initial_time,
            "max_num_time_steps": 1,
        }
    dataset = get_dataset(
        dataset=dataset,
        which="test",
        num_trajectories=1,
        data_path=data_path,
        move_to_local_scratch=None,
        **dataset_kwargs,
    )
    return dataset

def get_trainer(
    model_path,
    batch_size,
    dataset,
    full_data=False,
    output_all_steps=False,
    workers=-1,
):
    """
    Get a trainer for the model (actually just using the interface for inference).

    Args:
        model_path: str
            Path to the model.
        batch_size: int
            Batch size for evaluation.
        dataset: BaseTimeDataset
            Test set.
        full_data: bool
            Whether to save the full data distribution.
        output_all_steps: bool
            Whether to output all preliminary steps in autoregressive rollout.
        workers: int
            Number of workers for evaluation. If -1 will use all available cores.
    """
    num_cpu_cores = len(psutil.Process().cpu_affinity())
    if workers == -1:
        workers = num_cpu_cores
    if workers > num_cpu_cores:
        workers = num_cpu_cores
    assert workers > 0

    model = ScOT.from_pretrained(model_path)
    args = TrainingArguments(
        output_dir=".",
        per_device_eval_batch_size=batch_size,
        eval_accumulation_steps=16,
        dataloader_num_workers=workers,
        log_level="info"
    )
    time_involved = isinstance(dataset, BaseTimeDataset)

    def compute_metrics(eval_preds):
        if time_involved and output_all_steps:
            return {}
        channel_list = dataset.channel_slice_list

        def get_relative_statistics(errors):
            median_error = np.median(errors, axis=0)
            mean_error = np.mean(errors, axis=0)
            std_error = np.std(errors, axis=0)
            min_error = np.min(errors, axis=0)
            max_error = np.max(errors, axis=0)
            return {
                "median_relative_l1_error": median_error,
                "mean_relative_l1_error": mean_error,
                "std_relative_l1_error": std_error,
                "min_relative_l1_error": min_error,
                "max_relative_l1_error": max_error,
            }

        def get_statistics(errors):
            median_error = np.median(errors, axis=0)
            mean_error = np.mean(errors, axis=0)
            std_error = np.std(errors, axis=0)
            min_error = np.min(errors, axis=0)
            max_error = np.max(errors, axis=0)
            return {
                "median_l1_error": median_error,
                "mean_l1_error": mean_error,
                "std_l1_error": std_error,
                "min_l1_error": min_error,
                "max_l1_error": max_error,
            }

        relative_errors = [
            relative_lp_error(
                eval_preds.predictions[:, channel_list[i] : channel_list[i + 1]],
                eval_preds.label_ids[:, channel_list[i] : channel_list[i + 1]],
                p=1,
                return_percent=True,
            )
            for i in range(len(channel_list) - 1)
        ]

        errors = [
            lp_error(
                eval_preds.predictions[:, channel_list[i] : channel_list[i + 1]],
                eval_preds.label_ids[:, channel_list[i] : channel_list[i + 1]],
                p=1,
            )
            for i in range(len(channel_list) - 1)
        ]

        relative_error_statistics = [
            get_relative_statistics(relative_errors[i])
            for i in range(len(channel_list) - 1)
        ]

        error_statistics = [
            get_statistics(errors[i]) for i in range(len(channel_list) - 1)
        ]

        if dataset.output_dim == 1:
            relative_error_statistics = relative_error_statistics[0]
            error_statistics = error_statistics[0]
            if full_data:
                relative_error_statistics["relative_full_data"] = relative_errors[
                    0
                ].tolist()
                error_statistics["full_data"] = errors[0].tolist()
            return {**relative_error_statistics, **error_statistics}
        else:
            mean_over_relative_means = np.mean(
                np.array(
                    [
                        stats["mean_relative_l1_error"]
                        for stats in relative_error_statistics
                    ]
                ),
                axis=0,
            )
            mean_over_relative_medians = np.mean(
                np.array(
                    [
                        stats["median_relative_l1_error"]
                        for stats in relative_error_statistics
                    ]
                ),
                axis=0,
            )
            mean_over_means = np.mean(
                np.array([stats["mean_l1_error"] for stats in error_statistics]), axis=0
            )
            mean_over_medians = np.mean(
                np.array([stats["median_l1_error"] for stats in error_statistics]),
                axis=0,
            )

            error_statistics_ = {
                "mean_relative_l1_error": mean_over_relative_means,
                "mean_over_median_relative_l1_error": mean_over_relative_medians,
                "mean_l1_error": mean_over_means,
                "mean_over_median_l1_error": mean_over_medians,
            }
            #!! The above is different from train and finetune (here mean_relative_l1_error is mean over medians instead of mean over means)
            for i, stats in enumerate(relative_error_statistics):
                for key, value in stats.items():
                    error_statistics_[
                        dataset.printable_channel_description[i] + "/" + key
                    ] = value
                    if full_data:
                        error_statistics_[
                            dataset.printable_channel_description[i]
                            + "/"
                            + "relative_full_data"
                        ] = relative_errors[i].tolist()
            for i, stats in enumerate(error_statistics):
                for key, value in stats.items():
                    error_statistics_[
                        dataset.printable_channel_description[i] + "/" + key
                    ] = value
                    if full_data:
                        error_statistics_[
                            dataset.printable_channel_description[i] + "/" + "full_data"
                        ] = errors[i].tolist()
            return error_statistics_

    trainer = Trainer(
        model=model,
        args=args,
        compute_metrics=compute_metrics,
    )
    return trainer

def rollout(trainer, dataset, ar_steps=1, output_all_steps=False):
    """
    Do a rollout of the model.

    Args:
        trainer: Trainer
            Trainer for the model.
        dataset: BaseTimeDataset
            Test set.
        ar_steps: int or list
            Number of autoregressive steps to take. A single int n is interpreted as taking n homogeneous steps, a list of ints [j_0, j_1, ...] is interpreted as taking a step of size j_i.
        output_all_steps: bool
            Whether to output all preliminary steps in autoregressive rollout.
    """
    time_involved = isinstance(dataset, BaseTimeDataset)
    if time_involved and ar_steps != 1:
        trainer.set_ar_steps(ar_steps, output_all_steps=output_all_steps)
    else:
        trainer.set_ar_steps(ar_steps=1, output_all_steps=False)

    prediction = trainer.predict(dataset, metric_key_prefix="")

    try:
        return prediction.predictions, prediction.label_ids, prediction.metrics
    except:
        return prediction.predictions

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    params = read_parser(parser).parse_args()

    ar_steps = params.ar_steps #TODO look into it for not None cases.

    dataset_kwargs = {}

    dataset = get_test_set(
                params.dataset,
                params.data_path,
                params.initial_time,
                params.final_time,
                dataset_kwargs,
            )
    trainer = get_trainer(
                params.model_path,
                params.batch_size,
                dataset,
                full_data=params.full_data,
            )
    predictions, label_ids, metrics = rollout(
                trainer,
                dataset,
                ar_steps=params.ar_steps,
                output_all_steps=False,
            )
    data = {
                "dataset": params.dataset,
                "initial_time": params.initial_time,
                "final_time": params.final_time,
                "ar_steps": ar_steps,
                **metrics,
            }
    
    inputs = dataset[0]["pixel_values"]

    print("Inference is on ...")

    create_predictions_plot_infer(
            inputs,
            predictions,
            label_ids,
            **data
        )

