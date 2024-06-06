#export LD_LIBRARY_PATH=/usr/local/cuda-10.1/lib64:$LD_LIBRARY_PATH
python sample_data_loader.py --node 0 --nodesize 1 --ngpus 4 --disturl tcp://127.0.0.1:12345