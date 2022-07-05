. ./activate.sh

for fold in {0..4}; do
    for seed in {0..4}; do
        echo "Seed: $seed, Fold: $fold"
        python_cmd="WANDB_ENABLED=TRUE bash main.sh configs/m1m2_planar.yml --train --seed=$seed --dataset.fold=$fold --trainer.earlystopping.patience=128"
        launch_cmd="sbatch -p shared -t 24:00:00 --wrap '$python_cmd'"
        echo $launch_cmd
        eval $launch_cmd
    done
done
