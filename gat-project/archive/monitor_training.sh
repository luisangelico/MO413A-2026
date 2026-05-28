#!/bin/bash
echo "🔍 GAT Training Monitor"
echo "======================="
echo ""
echo "Training Process:"
ps aux | grep train_model.py | grep -v grep | awk '{print "  PID:", $2, "| CPU:", $3"%", "| Memory:", $4"%"}'
echo ""
echo "Latest Run Directory:"
ls -dt data/processed/run_* 2>/dev/null | head -1
echo ""
echo "Checkpoint Files:"
ls -lh $(ls -dt data/processed/run_* 2>/dev/null | head -1)/*.pt 2>/dev/null || echo "  No checkpoints yet (saves every 100 epochs)"
echo ""
echo "Press Ctrl+C to stop monitoring"
echo ""
while true; do
    sleep 10
    clear
    echo "🔍 GAT Training Monitor - $(date '+%H:%M:%S')"
    echo "======================="
    echo ""
    ps aux | grep train_model.py | grep -v grep | awk '{print "  Process active - CPU:", $3"%", "| Memory:", $4"%"}' || echo "  Training completed or stopped"
    echo ""
    latest_run=$(ls -dt data/processed/run_* 2>/dev/null | head -1)
    echo "Latest Run: $latest_run"
    checkpoints=$(ls $latest_run/*.pt 2>/dev/null | wc -l)
    echo "Checkpoints: $checkpoints"
    if [ $checkpoints -gt 0 ]; then
        echo ""
        echo "Latest checkpoint:"
        ls -lht $latest_run/*.pt | head -1
    fi
done
