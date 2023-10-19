cd ../..


# # # # # # # # # -----------------------------------------------------------
 CUDA_VISIBLE_DEVICES=0,1 python tools/test.py \
     --cfg experiments/coco/hrnet/w48_384x288_adam_lr1e-3.yaml \
     GPUS '(0,1)' \
     OUTPUT_DIR 'outputs/gt_noise_CoAM/'\
     LOG_DIR 'logs/gt_noise_CoAM/'\
     DATASET.DATASET 'ochuman' \
     DATASET.TEST_IMAGE_DIR 'your_image_dir'\
     DATASET.TEST_ANNOTATION_FILE 'your_annotation_path' \
     DATASET.COLORED 'False' \
     DATASET.STACKED_CONDITION 'True' \
     DATASET.BU_BBOX_MARGIN 0 \
     TEST.BATCH_SIZE_PER_GPU 12 \
     TEST.USE_GT_BBOX False \
     TEST.USE_BU_BBOX True \
     TEST.FLIP_TEST True \
     TEST.MODEL_FILE 'models/gt_noise_COAM_iccv_final_state.pth' \
     TEST.COCO_BBOX_FILE 'your_bu/buctd_prediction' \
     MODEL.NAME 'pose_hrnet_coam' \
     MODEL.EXTRA.USE_ATTENTION True \
     MODEL.ATT_MODULES '[False, True, False, False]' \
     MODEL.ATT_CHANNEL_ONLY False \
     MODEL.ATTENTION_HEADS 1 \
     MODEL.CONDITIONAL_TOPDOWN True 
