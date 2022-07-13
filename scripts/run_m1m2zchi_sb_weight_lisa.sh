. ./activate.sh

experiment_name="m1m2zchi_sb_weight"
for sb_weight in 0 1 0.5 0.1 0.00001; do
    python_cmd="WANDB_ENABLED=TRUE sh main.sh configs/m1m2zchi_sb_planar.yml --train \
        --model.sb_weight=$sb_weight \
        --experiment.name=$experiment_name"_"$sb_weight \
        --experiment.group=$experiment_name"
    launch_cmd="sbatch -p shared -c 8 -t 24:00:00 --wrap '$python_cmd'"
    echo $launch_cmd
done
