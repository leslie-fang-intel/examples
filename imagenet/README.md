
## build pytorch
  ```
  conda install numpy ninja pyyaml mkl mkl-include setuptools cmake cffi
  git clone https://github.com/pytorch/pytorch.git  
  cd pytorch
  python setup.py clean
  git submodule sync &&  git submodule update --init --recursive
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
 
## running on CPX with MKLNDN backend
  ```
  export LD_PRELOAD=/path/to/jemalloc/lib/libjemalloc.so
  export MALLOC_CONF="oversize_threshold:1,background_thread:true,metadata_thp:auto,dirty_decay_ms:9000000000,muzzy_decay_ms:9000000000"
  bash run_inference_cpu_multi_instance.sh mkldnn
  ```

## running on CPX with no MKLNDN
  ```
  bash run_inference_cpu_multi_instance.sh
  ```
