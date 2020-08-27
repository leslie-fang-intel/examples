##############################################################################
#### 1) int8 calibration step(non fusion path using ipex):
####    bash run_int8_accuracy_ipex.sh resnet50 DATA_PATH dnnl int8 calibration
#### 2) int8 inference step(none fusion path using ipex):
####    bash run_int8_accuracy_ipex.sh resnet50 DATA_PATH dnnl int8
#### 3) fp32 inference step(non fusion path using ipex):
####    bash run_int8_accuracy_ipex.sh resnet50 DATA_PATH dnnl
###############################################################################
export DNNL_PRIMITIVE_CACHE_CAPACITY=1024

BATCH_SIZE=128

ARGS=""
if [[ "$1" == "resnet50" ]]
then
    ARGS="$ARGS resnet50"
    echo "### running resnet50 model"
else
    ARGS="$ARGS resnext101_32x4d"
    echo "### running resnext101_32x4d model"
fi

ARGS="$ARGS $2"
echo "### dataset path: $2"

if [[ "$3" == "dnnl" ]]
then
    ARGS="$ARGS --dnnl"
    echo "### running auto_dnnl mode"
fi

if [[ "$4" == "int8" ]]
then
    ARGS="$ARGS --int8"
    echo "### running int8 datatype"
fi

if [[ "$5" == "calibration" ]]
then
    BATCH_SIZE=32
    ARGS="$ARGS --calibration"
    echo "### running int8 calibration"
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

if [ "$1" == "resnet50" ]; then
  python -u main.py -e -a $ARGS --ipex --pretrained -j 0 -b $BATCH_SIZE --configure-dir resnet50_configure.json
else
  python -u main.py -e -a $ARGS --ipex --pretrained -j 0 -b $BATCH_SIZE --checkpoint-dir checkpoints/resnext101_32x4d/checkpoint.pth.tar --configure-dir resnext101_configure.json
fi
