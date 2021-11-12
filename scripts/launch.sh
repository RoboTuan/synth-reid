#!/bin/bash

# Run this script from the root directory, example:
# ./scripts/launch.sh -c configs/im_bnneck_triplet_val_multi.yaml -g 3 -t bnneck_tri -n Y -a Y
new_session="N"
tmux_attach="N"
while getopts c:g:t:n:a: flag
do
    case "${flag}" in
        c) configuration=${OPTARG};;
        g) gpu_id=${OPTARG};;
        t) tmux_session=${OPTARG};;
        n) new_session=${OPTARG};;
        a) tmux_attach=${OPTARG};;
    esac
done

nvidia-smi
echo "Running configuration: "$configuration
echo "GPU id: "$gpu_id
echo "tmux session: "$tmux_session

tmux has-session -t $tmux_session 2>/dev/null

if [ $? != 0 ]; then
    echo "No session "$tmux_session
    if [ $new_session = "Y" ]; then
        echo "New nession "$tmux_session
        tmux new-session -d -s $tmux_session
        tmux send-keys -t $tmux_session "CUBLAS_WORKSPACE_CONFIG=:16:8 CUDA_VISIBLE_DEVICES=$gpu_id python scripts/main.py --config-file $configuration" Enter
        if [ $tmux_attach = "Y" ]; then
            tmux attach -t $tmux_session
        elif [ $tmux_attach = "N" ]; then
            :  # do nothing
        else
            echo "Option -a must be 'Y'/'N', '"$tmux_attach"' is not correct, exiting..."
            exit 1
        fi
    elif [ $new_session = "N" ]; then
        echo "New nession "$tmux_session "was not created, exiting and killing session..."
        tmux kill-session -t $tmux_session
        exit 1
    else
        echo "Option -n must be 'Y'/'N', '"$new_session"' is not correct, exiting..."
        exit 1
    fi
else
    echo "Existing session "$tmux_session
    tmux send-keys -t $tmux_session "CUBLAS_WORKSPACE_CONFIG=:16:8 CUDA_VISIBLE_DEVICES=$gpu_id python scripts/main.py --config-file $configuration" Enter
    if [ $tmux_attach = "Y" ]; then
        tmux attach -t $tmux_session
    elif [ $tmux_attach = "N" ]; then
        :  # do nothing
    else
        echo "Option -a must be 'Y'/'N', '"$tmux_attach"' is not correct, the process is continuing in the tmux session..."
        exit 1
    fi
fi

