cd ../..


# # # # # # # # # -----------------------------------------------------------
 CUDA_VISIBLE_DEVICES=0,1 python tools/test.py \
     --cfg experiments/coco/hrnet/w48_384x288_adam_lr1e-3.yaml \
     GPUS '(0,1)' \
     OUTPUT_DIR 'outputs/gt_noise_prenet'\
     LOG_DIR 'logs/gt_noise_prenet'\
     DATASET.DATASET 'coco' \
     DATASET.TEST_IMAGE_DIR 'data/coco/images'\
     DATASET.TEST_ANNOTATION_FILE 'data/annotations/person_keypoints_val2017.json' \
     DATASET.COLORED 'True' \
     DATASET.BU_BBOX_MARGIN 0 \
     TRAIN.LR 0.002 \
     TRAIN.BEGIN_EPOCH 0 \
     TRAIN.END_EPOCH 110 \
     TRAIN.LR_STEP '(70, 100)' \
     TRAIN.BATCH_SIZE_PER_GPU 12 \
     TRAIN.USE_BU_BBOX True \
     TEST.BATCH_SIZE_PER_GPU 12 \
     TEST.USE_GT_BBOX False \
     TEST.USE_BU_BBOX True \
     TEST.FLIP_TEST True \
     TEST.MODEL_FILE 'models/COCO-BUCTD-preNet-W48.pth' \
     TEST.COCO_BBOX_FILE './PETR/results/keypoints_test_results.json' \
     EPOCH_EVAL_FREQ 10 \
     PRINT_FREQ 100 \
     MODEL.NAME 'pose_hrnet' \
     MODEL.EXTRA.USE_PRE_NET True \
     MODEL.CONDITIONAL_TOPDOWN True