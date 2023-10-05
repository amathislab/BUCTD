cd ../..

# # # # # # # # # -----------------------------------------------------------
CUDA_VISIBLE_DEVICES=1 python tools/train.py \
     --cfg experiments/coco/hrnet/w48_384x288_adam_lr1e-3.yaml \
     GPUS '(1,)' \
     OUTPUT_DIR 'runs/models/transpose'\
     LOG_DIR 'runs/log/transpose'\
     DATASET.DATASET 'coco' \
     DATASET.TRAIN_IMAGE_DIR 'data/coco/images'\
     DATASET.TRAIN_ANNOTATION_FILE 'data/coco/annotations/train_cond.json' \
     DATASET.TEST_IMAGE_DIR 'data/coco/images'\
     DATASET.TEST_ANNOTATION_FILE 'data/coco/annotations/test.json' \
     DATASET.COLORED True \
     TRAIN.BATCH_SIZE_PER_GPU 32 \
     TRAIN.USE_BU_BBOX True \
     TEST.BATCH_SIZE_PER_GPU 32 \
     TEST.FLIP_TEST False \
     TEST.USE_BU_BBOX True \
     EPOCH_EVAL_FREQ 1 \
     TEST.COCO_BBOX_FILE 'prediction_file_from_other_model' \
     MODEL.NAME 'transpose_h' \
     MODEL.EXTRA.USE_ATTENTION True \
     MODEL.CONDITIONAL_TOPDOWN True
     