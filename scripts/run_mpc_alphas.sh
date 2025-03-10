#!/bin/bash

# Argument --> controller type

# Define the Python script path
GUESS_SCRIPT="guess.py"
MPC_SCRIPT="mpc.py"

# Define the output log file
LOG_GUESS="guess_sm_hard.txt"
LOG_MPC="mpc_sm_hard.txt"

# Clear the log file if it exists
> "$LOG_GUESS"
> "$LOG_MPC"

# Controller type
CONT_ARG=("naive" "zerovel" "st" "htwa" "receding" "real")

# Define the arguments for the Python script
ALPHA_ARG=("20" "30" "40") # "50") 

# Loop through the arguments
for CONT in "${CONT_ARG[@]}"; do
    # Controller
    echo "Running tasks with controller $CONT" | tee -a "$LOG_GUESS" "$LOG_MPC"
    echo "-----------------------------------" | tee -a "$LOG_GUESS" "$LOG_MPC"

    for ARG in "${ALPHA_ARG[@]}"; do
        # Guess
        echo "Running $GUESS_SCRIPT with argument alpha $ARG" | tee -a "$LOG_GUESS"
        nohup python "$GUESS_SCRIPT" -c="$CONT" --alpha="$ARG" >> "$LOG_GUESS" 2>&1
        echo "Completed execution" | tee -a "$LOG_GUESS"
        echo "----------------------------------------" | tee -a "$LOG_GUESS"

        # MPC
        echo "Running $MPC_SCRIPT with argument alpha $ARG" | tee -a "$LOG_MPC"
        nohup python "$MPC_SCRIPT" -c="$CONT" --alpha="$ARG" >> "$LOG_MPC" 2>&1
        echo "Completed execution" | tee -a "$LOG_MPC"
        echo "----------------------------------------" | tee -a "$LOG_MPC"
    done
done

echo "All executions completed. Log written to $LOG_GUESS" and "$LOG_MPC"
