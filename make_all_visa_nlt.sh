
#!/usr/bin/env bash
python convert_visa_to_mvtec_format.py \
    -s datasets/visa \
    -t datasets/visa_ \

for seed in {01..05}; do
  python make_mvtecad_nlt.py \
    --source-dir datasets/visa_ \
    --dest-dir   datasets/visa-step_k1-seed${seed} \
    --prune-manifest manifest/visa-nlt/step_k1/seed${seed}/prune_good.txt \
    --noisy-manifest manifest/visa-nlt/step_k1/seed${seed}/inject_defects.txt
done

for seed in {01..05}; do
  python make_mvtecad_nlt.py \
    --source-dir datasets/visa_ \
    --dest-dir   datasets/visa-step_k4-seed${seed} \
    --prune-manifest manifest/visa-nlt/step_k4/seed${seed}/prune_good.txt \
    --noisy-manifest manifest/visa-nlt/step_k4/seed${seed}/inject_defects.txt
done

for seed in {01..05}; do
  python make_mvtecad_nlt.py \
    --source-dir datasets/visa_ \
    --dest-dir   datasets/visa-pareto-seed${seed} \
    --prune-manifest manifest/visa-nlt/pareto/seed${seed}/prune_good.txt \
    --noisy-manifest manifest/visa-nlt/pareto/seed${seed}/inject_defects.txt
done