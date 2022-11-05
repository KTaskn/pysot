export PYTHONPATH=/path/to/pysot:$PYTHONPATH

INPUT=/video/fish.mp4
OUTPUT=/video/fish_tracking.mp4
X=500
Y=320
W=90
H=50

python tools/demo_without_window.py \
    --config experiments/siamrpn_r50_l234_dwxcorr/config.yaml \
    --snapshot experiments/siamrpn_r50_l234_dwxcorr/siamrpn_r50_l234_dwxcorr.pth \
    --input_video_name $INPUT \
    --output_video_name $OUTPUT \
    --init_rect $X $Y $W $H