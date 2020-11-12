##############################################################################
#### 1) MKLDNN model inference :
####    bash run_fp32_accuracy.sh resnet50/resnext101_32x8d mkldnn
#### 2) Vanilla cpu infernece path:
####    bash run_fp32_accuracy.sh resnet50/resnext101_32x8d mkldnn

###############################################################################
export DNNL_PRIMITIVE_CACHE_CAPACITY=1024

if [ -z "${DATASET_DIR}" ]; then
  echo "The required environment variable DATASET_DIR has not been set"
  exit 1
fi

if [ ! -d "${DATASET_DIR}" ]; then
  echo "The DATASET_DIR '${DATASET_DIR}' does not exist"
  exit 1
fi

BATCH_SIZE=128

ARGS=""
if [ "$1" == "resnet50" ]; then
    ARGS="$ARGS resnet50"
    echo "### running resnet50 model"
else
    ARGS="$ARGS resnext101_32x8d"
    echo "### running resnext101_32x8d model"
fi

if [ "$2" == "mkldnn" ]; then
    ARGS="$ARGS --mkldnn"
    echo "### running auto_dnnl mode"
fi


CORES=`lscpu | grep Core | awk '{print $4}'`
SOCKETS=`lscpu | grep Socket | awk '{print $2}'`
TOTAL_CORES=`expr $CORES \* $SOCKETS`

KMP_SETTING="KMP_AFFINITY=granularity=fine,compact,1,0"


export OMP_NUM_THREADS=$TOTAL_CORES
export $KMP_SETTING

echo -e "### using OMP_NUM_THREADS=$TOTAL_CORES"
echo -e "### using $KMP_SETTING\n\n"
sleep 3

python -u main.py -e -a $ARGS --data ${DATASET_DIR} --pretrained -j 0 -b $BATCH_SIZE
