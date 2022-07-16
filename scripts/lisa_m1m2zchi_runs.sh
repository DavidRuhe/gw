. ./activate.sh

experiment_name="m1m2zchi"

python_cmd="WANDB_ENABLED=TRUE sh main.sh configs/m1m2zchi_planar.yml --train \
    --experiment.name=$experiment_name"
launch_cmd="sbatch -p normal -t 24:00:00 --wrap '$python_cmd'"
echo $launch_cmd

python_cmd="WANDB_ENABLED=TRUE sh main.sh configs/m1m2zchi_sb_planar.yml --train \
    --experiment.name=$experiment_name \
    --model.prior_weight=0"
launch_cmd="sbatch -p normal -t 24:00:00 --wrap '$python_cmd'"
echo $launch_cmd

python_cmd="WANDB_ENABLED=TRUE sh main.sh configs/m1m2zchi_sb_planar.yml --train \
    --experiment.name=$experiment_name \
    --model.prior_weight=1"
launch_cmd="sbatch -p normal -t 24:00:00 --wrap '$python_cmd'"
echo $launch_cmd

python_cmd="WANDB_ENABLED=TRUE sh main.sh configs/m1m2zchi_sb_planar.yml --train \
    --experiment.name=$experiment_name \
    --trainer.scheduler.prior_weight=\[0,0,32,0.1,64,0.5,96,1\]"
launch_cmd="sbatch -p normal -t 24:00:00 --wrap '$python_cmd'"
echo $launch_cmd
