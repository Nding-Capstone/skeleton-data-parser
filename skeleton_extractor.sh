ffmpeg -ss $1 -i ./video/$2.mp4 -c copy -t $3 ./video/cut_$2.mp4
ffmpeg -i ./video/cut_$2.mp4 -r 4 ./test_in/$2_%04d.png
python extract_skeleton_vector.py --model 101 --image_dir ./test_in --output_dir ./test_out/$2