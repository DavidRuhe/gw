. ./activate.sh

experiment_name="m1m2zchi_sb"

python_cmd="WANDB_ENABLED=TRUE sh main.sh configs/m1m2zchi_sb_planar.yml --train \
    --experiment.name=$experiment_name_prior_0 \
    --model.prior_weight=0 \
    --trainer.scheduler.sb_weights=\[0, 0, 32, 0.1, 128, 0.5, 160, 1\]"
launch_cmd="sbatch -p normal -t 24:00:00 --wrap '$python_cmd'"
echo $launch_cmd

python_cmd="WANDB_ENABLED=TRUE sh main.sh configs/m1m2zchi_sb_planar.yml --train \
    --experiment.name=$experiment_name_prior_0 \
    --model.prior_weight=1 \
    --trainer.scheduler.sb_weights=\[0, 0, 32, 0.1, 128, 0.5, 160, 1\]"
launch_cmd="sbatch -p normal -t 24:00:00 --wrap '$python_cmd'"
echo $launch_cmd