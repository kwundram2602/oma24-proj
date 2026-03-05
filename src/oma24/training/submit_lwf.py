"""Command-line script to submit LWF-DLR U-Net training to SLURM."""

import argparse
import logging
from pathlib import Path

import submitit
import yaml

from training.lwf_unet_aspp_trainer import LWFUNetASPPTrainer
from training.lwf_unet_loss_trainer import LWFUNetLossTrainer
from training.lwf_unet_skeleton_trainer import LWFUNetSkeletonTrainer
from training.lwf_unet_trainer import LWFUNetTrainer

logger = logging.getLogger(__name__)


def submit_lwf_training(cfg: dict) -> submitit.Job:
    """Submit LWF-DLR U-Net training job to SLURM cluster.

    Parameters
    ----------
    cfg : dict
        Configuration dictionary loaded from a YAML file.

    Returns
    -------
    submitit.Job
        Submitted job object.
    """
    user = cfg["user"]
    root_hpc = Path("/dss/dsstbyfs02/pn49ci/pn49ci-dss-0026")
    user_path = root_hpc / user
    log_dir = user_path / "experiments/LWF-DLR/slurm_logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    executor = submitit.AutoExecutor(folder=str(log_dir))

    num_workers = cfg.get("num_workers", 16)
    cpus_per_task = 2 + num_workers

    clusters = cfg.get("clusters", "hpda2")
    account = cfg.get("account", "pn39sa-c")
    mail_user = cfg.get("mail_user", None)
    exclude_nodes = cfg.get("exclude_nodes", None)

    slurm_additional_parameters = {
        "clusters": clusters,
        "account": account,
    }

    if mail_user:
        slurm_additional_parameters["mail-type"] = "END"
        slurm_additional_parameters["mail-user"] = mail_user

    if exclude_nodes:
        slurm_additional_parameters["exclude"] = exclude_nodes

    venv_path = Path(cfg.get("venv_path", str(root_hpc / user / ".venv")))
    time_str = cfg.get("time", "03:00:00")

    executor.update_parameters(
        slurm_partition=cfg.get("partition", "hpda2_compute_gpu"),
        timeout_min=int(time_str.split(":")[0]) * 60 + int(time_str.split(":")[1]),
        gpus_per_node=cfg.get("gpus_per_node", 1),
        cpus_per_task=cpus_per_task,
        mem_gb=cfg.get("mem_gb", 256),
        slurm_job_name=f"lwf_unet_{cfg.get('experiment_id', 'run')}",
        slurm_additional_parameters=slurm_additional_parameters,
        slurm_setup=[
            "module load slurm_setup",
            f"source {venv_path}/bin/activate",
        ],
    )

    # Shared trainer kwargs
    common = dict(
        user=user,
        csv_name=cfg.get("csv_name", "dev_1000_file.csv"),
        epochs=cfg.get("epochs", 32),
        batch_size=cfg.get("batch_size", 12),
        num_workers=num_workers,
        prefetch_factor=cfg.get("prefetch_factor", 8),
        experiment_id=cfg.get("experiment_id", "baseline_unet_slurm"),
    )

    use_skeleton_trainer = cfg.get("use_skeleton_trainer", False)
    use_aspp_trainer = cfg.get("use_aspp_trainer", False)
    use_loss_trainer = cfg.get("use_loss_trainer", False)

    class_weights = cfg.get("class_weights", [1.0, 50.0, 5.0])
    use_class_weights = cfg.get("use_class_weights", True)
    use_dice_loss = cfg.get("use_dice_loss", True)
    dice_loss_weight = cfg.get("dice_loss_weight", 0.3)
    use_residual = cfg.get("use_residual", True)
    use_aspp = cfg.get("use_aspp", True)
    aspp_rates = tuple(cfg.get("aspp_rates", [3, 6, 9, 12]))

    if use_skeleton_trainer:
        trainer = LWFUNetSkeletonTrainer(
            **common,
            use_class_weights=use_class_weights,
            class_weights=class_weights,
            use_dice_loss=use_dice_loss,
            dice_loss_weight=dice_loss_weight,
            use_residual=use_residual,
            use_aspp=use_aspp,
            aspp_rates=aspp_rates,
            use_distance=cfg.get("use_distance", False),
            distance_max=cfg.get("distance_max", 128),
            use_input_skeleton=cfg.get("use_input_skeleton", False),
            use_dual_head=cfg.get("use_dual_head", False),
            skeleton_class=cfg.get("skeleton_class", 1),
            skeleton_loss_weight=cfg.get("skeleton_loss_weight", 1.0),
        )
    elif use_aspp_trainer:
        trainer = LWFUNetASPPTrainer(
            **common,
            use_class_weights=use_class_weights,
            class_weights=class_weights,
            use_dice_loss=use_dice_loss,
            dice_loss_weight=dice_loss_weight,
            use_residual=use_residual,
            use_aspp=use_aspp,
            aspp_rates=aspp_rates,
        )
    elif use_loss_trainer:
        trainer = LWFUNetLossTrainer(
            **common,
            class_weights=class_weights,
            dice_loss_weight=dice_loss_weight,
        )
    else:
        trainer = LWFUNetTrainer(**common)

    job = executor.submit(trainer)

    logger.info("Job submitted with ID: %s", job.job_id)
    logger.info("SLURM log directory: %s", log_dir)
    logger.info("Experiment directory: %s/experiments/LWF-DLR", user_path)

    return job


def main():  # noqa: D103
    parser = argparse.ArgumentParser(
        description="Submit LWF-DLR U-Net segmentation training to SLURM cluster"
    )
    parser.add_argument(
        "config",
        type=Path,
        help="Path to the YAML configuration file",
    )
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    job = submit_lwf_training(cfg)

    logger.info("Job submitted successfully!")
    logger.info("Job ID: %s", job.job_id)
    logger.info("To check job status: squeue -j %s", job.job_id)
    logger.info("To cancel job: scancel %s", job.job_id)


if __name__ == "__main__":
    main()
