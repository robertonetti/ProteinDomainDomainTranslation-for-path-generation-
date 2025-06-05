#!/usr/bin/env bash
# Salva come run_trainings.sh e rendilo eseguibile (chmod +x run_trainings.sh)

LOGFILE="$(dirname "$0")/run_trainings.log"

# ---- Self-detach: solo la PRIMA volta ----
if [[ -z "$DETACHED" ]]; then
  echo "$(date '+%Y-%m-%d %H:%M:%S') → Detaching and launching in background (logs → $LOGFILE)..."
  # esporta DETACHED=1 per la copia in background
  DETACHED=1 nohup "$0" "$@" > "$LOGFILE" 2>&1 &
  exit 0
fi

# ---- Da qui in poi siamo già in background! ----
echo "$(date '+%Y-%m-%d %H:%M:%S') → Starting trainings..."

# ---- Definizione degli array paralleli ----
trainsets=(
  "data/data/PF00014/evo_timescales/data10e3/condition_order/csv_format/combined10e3_train.csv"
  "data/data/PF00014/evo_timescales/data10e4/condition_order/csv_format/combined10e4_train.csv"
  "data/data/PF00014/evo_timescales/data10e5/condition_order/csv_format/combined10e5_train.csv"
  "data/data/PF00014/evo_timescales/data10e6/condition_order/csv_format/combined10e6_train.csv"
  "data/data/PF00014/evo_timescales/data10e7/condition_order/csv_format/combined10e7_train.csv"
)

valsets=(
  "data/data/PF00014/evo_timescales/data10e3/condition_order/csv_format/combined10e3_test.csv"
  "data/data/PF00014/evo_timescales/data10e4/condition_order/csv_format/combined10e4_test.csv"
  "data/data/PF00014/evo_timescales/data10e5/condition_order/csv_format/combined10e5_test.csv"
  "data/data/PF00014/evo_timescales/data10e6/condition_order/csv_format/combined10e6_test.csv"
  "data/data/PF00014/evo_timescales/data10e7/condition_order/csv_format/combined10e7_test.csv"
)

save_dirs=(
  "models/evo_timescales/compare_arDCA/evo_timescale_supershallow_traintested10e3_noPosMask_Mut"
  "models/evo_timescales/compare_arDCA/evo_timescale_supershallow_traintested10e4_noPosMask_Mut"
  "models/evo_timescales/compare_arDCA/evo_timescale_supershallow_traintested10e5_noPosMask_Mut"
  "models/evo_timescales/compare_arDCA/evo_timescale_supershallow_traintested10e6_noPosMask_Mut"
  "models/evo_timescales/compare_arDCA/evo_timescale_supershallow_traintested10e7_noPosMask_Mut"
)

run_names=(
  "evo_timescale_supershallow_traintested10e3_noPosMask_Mut"
  "evo_timescale_supershallow_traintested10e4_noPosMask_Mut"
  "evo_timescale_supershallow_traintested10e5_noPosMask_Mut"
  "evo_timescale_supershallow_traintested10e6_noPosMask_Mut"
  "evo_timescale_supershallow_traintested10e7_noPosMask_Mut"
)

# ---- Controllo lunghezza array ----
if [ "${#trainsets[@]}" -ne "${#valsets[@]}" ] || \
   [ "${#trainsets[@]}" -ne "${#save_dirs[@]}" ] || \
   [ "${#trainsets[@]}" -ne "${#run_names[@]}" ]; then
  echo "Errore: gli array trainsets, valsets, save_dirs e run_names devono avere la stessa lunghezza."
  exit 1
fi

# ---- Loop di training ----
for i in "${!trainsets[@]}"; do
    trainset="${trainsets[$i]}"
    valset="${valsets[$i]}"
    save_dir="${save_dirs[$i]}"
    run_name="${run_names[$i]}"

    echo "--------------------------------------------"
    echo "Esecuzione training #$((i+1)):"
    echo "  TRAINSET:   $trainset"
    echo "  VALSET:     $valset"
    echo "  SAVE DIR:   $save_dir"
    echo "  RUN NAME:   $run_name"
    echo "--------------------------------------------"

    python3 -m train \
      --trainset   "$trainset" \
      --valset     "$valset" \
      --save       "$save_dir" \
      --load       "" \
      --modelconfig "configs/super.shallow.config.json" \
      --outputfile "output.txt" \
      --run_name   "$run_name"
done

echo "Tutti i trainings sono terminati."
