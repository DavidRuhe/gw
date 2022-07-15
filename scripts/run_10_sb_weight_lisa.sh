. ./activate.sh

experiment_name="m1m2zchi_10_sb_weight"
for sb_weight in 1 0.1 0.01 0.001 0.0001 0.00001 0.000001 0.0000001 0.00000001 0.000000001; do
    python_cmd="WANDB_ENABLED=TRUE bash main.sh configs/m1m2_sb_planar.yml --train \
        --trainer.min_epochs=256 \
        --model.sb_weight=$sb_weight \
        --model.prior_weight=1 \
        --experiment.name=$experiment_name"_"$sb_weight \
        --experiment.group=$experiment_name"
    launch_cmd="sbatch -p shared -t 24:00:00 --wrap '$python_cmd'"
    echo $launch_cmd
done
