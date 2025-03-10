#!/bin/bash

# Argument --> controller type

# Define the Python script path
GUESS_SCRIPT="guess.py"
MPC_SCRIPT="mpc.py"

# Define the output log file
LOG_GUESS="guess_hor_hard.txt"
LOG_MPC="mpc_hor_hard.txt"

# Clear the log file if it exists
> "$LOG_GUESS"
> "$LOG_MPC"

# Controller type
CONT_ARG=("naive" "zerovel" "st" "htwa" "receding" "real")

# Define the arguments for the Python script
HORIZON_ARG=("20" "25" "30" "35" "40" "45") 

# Loop through the arguments
for CONT in "${CONT_ARG[@]}"; do
    # Controller
    echo "Running tasks with controller $CONT" | tee -a "$LOG_GUESS" "$LOG_MPC"
    echo "-----------------------------------" | tee -a "$LOG_GUESS" "$LOG_MPC"

    for ARG in "${HORIZON_ARG[@]}"; do
        # Guess
        echo "Running $GUESS_SCRIPT with argument horizon $ARG" | tee -a "$LOG_GUESS"
        nohup python "$GUESS_SCRIPT" -c="$CONT" --horizon="$ARG" >> "$LOG_GUESS" 2>&1
        echo "Completed execution" | tee -a "$LOG_GUESS"
        echo "----------------------------------------" | tee -a "$LOG_GUESS"

        # MPC
        echo "Running $MPC_SCRIPT with argument horizon $ARG" | tee -a "$LOG_MPC"
        nohup python "$MPC_SCRIPT" -c="$CONT" --horizon="$ARG" >> "$LOG_MPC" 2>&1
        echo "Completed execution" | tee -a "$LOG_MPC"
        echo "----------------------------------------" | tee -a "$LOG_MPC"
    done
done

echo "All executions completed. Log written to $LOG_GUESS" and "$LOG_MPC"
