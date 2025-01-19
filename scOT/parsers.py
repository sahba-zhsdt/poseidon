def read_parser(parser):

    parser.add_argument(
        "--model_path",
        type=str,
        required=False,
        help="Model path. Not required when mode==eval_sweep or save_samples_sweep.",
    )
    parser.add_argument(
        "--file",
        type=str,
        required=False,
        help="File to load/write to. May also be a directory to save samples.",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Path to data.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        help="Which test set to load. Not required if mode==eval_sweep or save_samples_sweep.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=5,
        help="Batch size for evaluation.",
    )
    parser.add_argument(
        "--full_data",
        action="store_true",
        help="Whether to save full data distributions.",
    )
    parser.add_argument(
        "--initial_time",
        type=int,
        default=None,
        help="Initial time step to start from.",
    )
    parser.add_argument(
        "--final_time",
        type=int,
        default=None,
        help="Final time step to end at.",
    )
    # parser.add_argument(
    #     "--ar_steps",
    #     type=int,
    #     nargs="+",
    #     default=None,
    #     help="Number of autoregressive steps to take. A single int n is interpreted as taking n homogeneous steps, a list of ints [j_0, j_1, ...] is interpreted as taking a step of size j_i.",
    # )
    parser.add_argument(
        "--ar_steps",
        type=int,
        default=None,
        help="Number of autoregressive steps to take. A single int n is interpreted as taking n homogeneous steps, a list of ints [j_0, j_1, ...] is interpreted as taking a step of size j_i.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=[
            "save_samples",
            "save_samples_sweep",
            "eval",
            "eval_sweep",
            "eval_accumulation_error",
            "eval_resolutions",
        ],
        default="eval",
        help="Mode to run. Can be either save_samples to save n samples, save_samples_sweep, eval (to evaluate a single model), eval_sweep (to evaluate all models in a wandb sweep), eval_accumulation_error (to evaluate a model's accumulation error), eval_resolutions (to evaluate a model on different resolutions).",
    )
    parser.add_argument(
        "--save_n_samples",
        type=int,
        default=1,
        help="Number of samples to save. Only required for mode==save_samples or save_samples_sweep.",
    )
    parser.add_argument(
        "--resolutions",
        type=int,
        nargs="+",
        help="List of resolutions to evaluate. Only required for mode==eval_resolutions.",
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="scOT",
        help="Wandb project name. Required if mode==eval_sweep or save_samples_sweep.",
    )
    parser.add_argument(
        "--wandb_entity",
        type=str,
        required=False,
        help="Wandb entity name. Required if mode==eval_sweep or save_samples_sweep.",
    )
    parser.add_argument(
        "--wandb_sweep_id",
        type=str,
        default=None,
        help="Wandb sweep id. Required if mode==eval_sweep or save_samples_sweep.",
    )
    parser.add_argument(
        "--ckpt_dir",
        type=str,
        required=True,
        help="Base checkpoint directory. Required if mode==eval_sweep or save_samples_sweep.",
    )
    parser.add_argument(
        "--exclude_dataset",
        type=str,
        nargs="+",
        default=[],
        help="Datasets to exclude from evaluation. Only relevant when mode==eval_sweep or save_samples_sweep.",
    )
    parser.add_argument(
        "--exclusively_evaluate_dataset",
        type=str,
        nargs="+",
        default=[],
        help="Datasets to exclusively evaluate. Only relevant when mode==eval_sweep or save_samples_sweep.",
    )
    parser.add_argument(
        "--just_velocities",
        action="store_true",
        help="Use just velocities in incompressible flow data.",
    )
    parser.add_argument(
        "--allow_failed",
        action="store_true",
        help="Allow failed runs to be taken into account with eval_sweep.",
    )
    parser.add_argument(
        "--append_time",
        action="store_true",
        help="Append .time to dataset name for evaluation.",
    )
    parser.add_argument(
        "--num_trajectories",
        type=int,
        default=128,
        help="Filter runs for number of training trajectories. Only relevant if mode==eval_sweep or save_samples_sweep.",
    )
    return parser