set -ex
# models
RESULTS_DIR='./results/test/western2manga'

# dataset
CLASS='color2manga'
MODEL='cycle_ganstft'
DIRECTION='BtoA' # from domain A to domain B
PREPROCESS='none'
LOAD_SIZE=512 # scale images to this size
CROP_SIZE=1024 # then crop to this size
INPUT_NC=1  # number of channels in the input image
OUTPUT_NC=3  # number of channels in the input image
NGF=48
NEF=48
NDF=32
NZ=64

# misc
GPU_ID=0   # gpu id
NUM_TEST=30 # number of input images duirng test
NUM_SAMPLES=1 # number of samples per input images
NAME=${CLASS}_${MODEL}

# command
CUDA_VISIBLE_DEVICES=${GPU_ID} \
python3 ./test.py \
  --dataroot ./datasets/${CLASS} \
  --results_dir ${RESULTS_DIR} \
  --checkpoints_dir ./checkpoints/${CLASS}/ \
  --name ${NAME} \
  --model ${MODEL} \
  --direction ${DIRECTION} \
  --preprocess ${PREPROCESS} \
  --load_size ${LOAD_SIZE} \
  --crop_size ${CROP_SIZE} \
  --input_nc ${INPUT_NC} \
  --output_nc ${OUTPUT_NC} \
  --nz ${NZ} \
  --netE conv_256 \
  --num_test ${NUM_TEST} \
  --n_samples ${NUM_SAMPLES} \
  --upsample bilinear \
  --ngf ${NGF} \
  --nef ${NEF} \
  --ndf ${NDF} \
  --center_crop \
  --color2screen \
  --no_flip
