. ./activate.sh

experiment_name="m1m2zchi_cuda"

for lr in 0.001 0.0005 0.0001; do
    for sb_weight in 1 0.1 0.01 0.001 0.0001; do
        python_cmd="WANDB_ENABLED=TRUE bash main.sh configs/m1m2zchi_sb_planar.yml --train \
            --trainer.min_epochs=512 \
            --model.sb_weight=$sb_weight \
            --model.lr=$lr \
            --experiment.name=$experiment_name"_"$sb_weight"_"$lr \
            --experiment.group=$experiment_name"
        launch_cmd="sbatch -p shared -t 24:00:00 --wrap '$python_cmd'"
        echo $launch_cmd
    done
done
