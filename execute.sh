export PYTHONPATH=/path/to/pysot:$PYTHONPATH
# python tools/demo_without_window.py \
#     --config experiments/siamrpn_r50_l234_dwxcorr/config.yaml \
#     --snapshot experiments/siamrpn_r50_l234_dwxcorr/siamrpn_r50_l234_dwxcorr.pth \
#     --input_video_name /video/01_017 \
#     --output_video_name ./01_017_tracking.mp4 \
#     --init_rect 60 35 90 130

python tools/demo_without_window.py \
    --config experiments/siamrpn_r50_l234_dwxcorr/config.yaml \
    --snapshot experiments/siamrpn_r50_l234_dwxcorr/siamrpn_r50_l234_dwxcorr.pth \
    --input_video_name /video/fish \
    --output_video_name ./fish_tracking.mp4 \
    --init_rect 500 320 90 50