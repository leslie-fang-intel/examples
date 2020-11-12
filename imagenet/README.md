
## build pytorch
  ```
  conda install numpy ninja pyyaml mkl mkl-include setuptools cmake cffi typing_extensions future six requests dataclasses
  git clone https://github.com/pytorch/pytorch.git  
  cd pytorch
  python setup.py clean
  git submodule sync &&  git submodule update --init --recursive
  # checkout Onednn to your test version or replace with your ideep(https://gitlab.devtools.intel.com/pytorch-cpu/ideep/-/tree/dnnl)
  cd third_party/ideep/mkl-dnn && git chckout master && git pull && git checkout v1.7
  export CMAKE_PREFIX_PATH=${CONDA_PREFIX:-"$(dirname $(which conda))/../"}
  python setup.py install
  ```
 
## buiild torchvision
  ```
  git clone https://github.com/pytorch/vision.git
  python setup.py install
  ```
 
## build jemalloc
  ```
  cd ..
  git clone  https://github.com/jemalloc/jemalloc.git    
  cd jemalloc 
  ./autogen.sh
  ./configure --prefix=your_path
  make
  make install
  ```

## Downloade PyTorch example
  ```
  git clone https://github.com/XiaobingSuper/examples.git
  git checkout for_onednn_upgrade
  ```

## running OneDNN model
  ```
  export LD_PRELOAD=/path/to/jemalloc/lib/libjemalloc.so
  export MALLOC_CONF="oversize_threshold:1,background_thread:true,metadata_thp:auto,dirty_decay_ms:9000000000,muzzy_decay_ms:9000000000"
  ```
  1. test accuracy:
  ```
  export DATASET_DIR= your imagenet path
  bash run_fp32_accuracy.sh resnet50/resnext101_32x8d mkldnn
  ```
  2. inference throughput(1socket/ins):
  ```
  bash run_inference_multi_instance_fp32.sh resnet50/resnext101_32x8d mkldnn
  ```
  3. inference realtime(4core/ins):
  ```
  bash run_inference_multi_instance_latency_fp32.sh resnet50/resnext101_32x8d mkldnn
  ```

## running vanilla cpu model
  ```
  export LD_PRELOAD=/path/to/jemalloc/lib/libjemalloc.so
  export MALLOC_CONF="oversize_threshold:1,background_thread:true,metadata_thp:auto,dirty_decay_ms:9000000000,muzzy_decay_ms:9000000000"
  ```
  1. test accuracy:
  ```
  export DATASET_DIR= your imagenet path
  bash run_fp32_accuracy.sh resnet50/resnext101_32x8d
  ```
  2. inference throughput(1socket/ins):
  ```
  bash run_inference_multi_instance_fp32.sh resnet50/resnext101_32x8d
  ```
  3. inference realtime(4core/ins):
  ```
  bash run_inference_multi_instance_latency_fp32.sh resnet50/resnext101_32x8d
  ```
