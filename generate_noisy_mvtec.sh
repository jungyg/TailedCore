seeds=(
    101
    102
    103
    104
    105
)

for seed in "${seeds[@]}"; do
    python generate_noisy_tailed_dataset.py --data_config "mvtec_pareto_random_nr10_seed${seed}" 
    python generate_noisy_tailed_dataset.py --data_config "mvtec_step_random_nr10_tk4_tr60_seed${seed}" 
    python generate_noisy_tailed_dataset.py --data_config "mvtec_step_random_nr10_tk1_tr60_seed${seed}" 
done

