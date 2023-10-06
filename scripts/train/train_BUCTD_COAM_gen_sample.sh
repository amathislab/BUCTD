cd ../..

# # # # # # # # # -----------------------------------------------------------
CUDA_VISIBLE_DEVICES=0 python tools/train.py \
     --cfg experiments/coco/hrnet/w48_384x288_adam_lr1e-3.yaml \
     GPUS '(0,)' \
     OUTPUT_DIR 'runs/models/CoAM_gtnoise'\
     LOG_DIR 'runs/log/CoAM_gtnoise'\
     DATASET.DATASET 'coco' \
     DATASET.TRAIN_IMAGE_DIR 'data/coco/images'\
     DATASET.TRAIN_ANNOTATION_FILE 'data/coco/annotations/train_cond.json' \
     DATASET.TEST_IMAGE_DIR 'data/coco/images'\
     DATASET.TEST_ANNOTATION_FILE 'data/coco/annotations/test.json' \
     DATASET.COLORED 'True' \
     DATASET.SYNTHESIS_POSE True \
     TRAIN.LR 0.002 \
     TRAIN.BEGIN_EPOCH 0 \
     TRAIN.END_EPOCH 110 \
     TRAIN.BATCH_SIZE_PER_GPU 32 \
     TRAIN.USE_BU_BBOX True \
     TEST.BATCH_SIZE_PER_GPU 32 \
     TEST.FLIP_TEST False \
     TEST.USE_BU_BBOX True \
     TEST.COCO_BBOX_FILE 'prediction_file_from_other_model' \
     EPOCH_EVAL_FREQ 1 \
     MODEL.NAME 'pose_hrnet_coam' \
     MODEL.EXTRA.USE_ATTENTION True \
     MODEL.ATT_MODULES '[False, True, False, False]' \
     MODEL.ATT_CHANNEL_ONLY False \
     MODEL.ATTENTION_HEADS 1 \
     MODEL.CONDITIONAL_TOPDOWN True
