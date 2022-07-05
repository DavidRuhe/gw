. ./activate.sh

for fold in {0..4}; do
    for seed in {0..4}; do
        echo "Seed: $seed, Fold: $fold"
        bash main.sh configs/m1_planar.yml --train --seed=$seed --dataset.fold=$fold --trainer.earlystopping.patience=128
    done
done
