seeds=(
    101
    102
    103
    104
    105
)

for seed in "${seeds[@]}"; do
    python generate_noisy_tailed_dataset.py --data_config "visa_pareto_random_nr05_seed${seed}" 
    python generate_noisy_tailed_dataset.py --data_config "visa_step_random_nr05_tk4_tr60_seed${seed}" 
    python generate_noisy_tailed_dataset.py --data_config "visa_step_random_nr05_tk1_tr60_seed${seed}" 
done

