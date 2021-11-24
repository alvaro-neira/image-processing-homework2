/content/yolov5
Downloading https://ultralytics.com/assets/Arial.ttf to /root/.config/Ultralytics/Arial.ttf...
train: weights=yolov5s.pt, cfg=, data=digits.yaml, hyp=data/hyps/hyp.scratch.yaml, epochs=10, batch_size=16, imgsz=640, rect=False, resume=False, nosave=False, noval=False, noautoanchor=False, evolve=None, bucket=, cache=None, image_weights=False, device=, multi_scale=False, single_cls=False, adam=False, sync_bn=False, workers=8, project=runs/train, name=exp, exist_ok=False, quad=False, linear_lr=False, label_smoothing=0.0, patience=100, freeze=0, save_period=-1, local_rank=-1, entity=None, upload_dataset=False, bbox_interval=-1, artifact_alias=latest
github: up to date with https://github.com/ultralytics/yolov5 ✅
YOLOv5 🚀 v6.0-103-g7a39803 torch 1.10.0+cu111 CUDA:0 (Tesla P100-PCIE-16GB, 16281MiB)

hyperparameters: lr0=0.01, lrf=0.1, momentum=0.937, weight_decay=0.0005, warmup_epochs=3.0, warmup_momentum=0.8, warmup_bias_lr=0.1, box=0.05, cls=0.5, cls_pw=1.0, obj=1.0, obj_pw=1.0, iou_t=0.2, anchor_t=4.0, fl_gamma=0.0, hsv_h=0.015, hsv_s=0.7, hsv_v=0.4, degrees=0.0, translate=0.1, scale=0.5, shear=0.0, perspective=0.0, flipud=0.0, fliplr=0.5, mosaic=1.0, mixup=0.0, copy_paste=0.0
Weights & Biases: run 'pip install wandb' to automatically track and visualize YOLOv5 🚀 runs (RECOMMENDED)
TensorBoard: Start with 'tensorboard --logdir runs/train', view at http://localhost:6006/
Downloading https://github.com/ultralytics/yolov5/releases/download/v6.0/yolov5s.pt to yolov5s.pt...
100% 14.0M/14.0M [00:00<00:00, 109MB/s] 

Overriding model.yaml nc=80 with nc=10

                 from  n    params  module                                  arguments                     
  0                -1  1      3520  models.common.Conv                      [3, 32, 6, 2, 2]              
  1                -1  1     18560  models.common.Conv                      [32, 64, 3, 2]                
  2                -1  1     18816  models.common.C3                        [64, 64, 1]                   
  3                -1  1     73984  models.common.Conv                      [64, 128, 3, 2]               
  4                -1  2    115712  models.common.C3                        [128, 128, 2]                 
  5                -1  1    295424  models.common.Conv                      [128, 256, 3, 2]              
  6                -1  3    625152  models.common.C3                        [256, 256, 3]                 
  7                -1  1   1180672  models.common.Conv                      [256, 512, 3, 2]              
  8                -1  1   1182720  models.common.C3                        [512, 512, 1]                 
  9                -1  1    656896  models.common.SPPF                      [512, 512, 5]                 
 10                -1  1    131584  models.common.Conv                      [512, 256, 1, 1]              
 11                -1  1         0  torch.nn.modules.upsampling.Upsample    [None, 2, 'nearest']          
 12           [-1, 6]  1         0  models.common.Concat                    [1]                           
 13                -1  1    361984  models.common.C3                        [512, 256, 1, False]          
 14                -1  1     33024  models.common.Conv                      [256, 128, 1, 1]              
 15                -1  1         0  torch.nn.modules.upsampling.Upsample    [None, 2, 'nearest']          
 16           [-1, 4]  1         0  models.common.Concat                    [1]                           
 17                -1  1     90880  models.common.C3                        [256, 128, 1, False]          
 18                -1  1    147712  models.common.Conv                      [128, 128, 3, 2]              
 19          [-1, 14]  1         0  models.common.Concat                    [1]                           
 20                -1  1    296448  models.common.C3                        [256, 256, 1, False]          
 21                -1  1    590336  models.common.Conv                      [256, 256, 3, 2]              
 22          [-1, 10]  1         0  models.common.Concat                    [1]                           
 23                -1  1   1182720  models.common.C3                        [512, 512, 1, False]          
 24      [17, 20, 23]  1     40455  models.yolo.Detect                      [10, [[10, 13, 16, 30, 33, 23], [30, 61, 62, 45, 59, 119], [116, 90, 156, 198, 373, 326]], [128, 256, 512]]
Model Summary: 270 layers, 7046599 parameters, 7046599 gradients, 15.9 GFLOPs

Transferred 343/349 items from yolov5s.pt
Scaled weight_decay = 0.0005
optimizer: SGD with parameter groups 57 weight, 60 weight (no decay), 60 bias
albumentations: version 1.0.3 required by YOLOv5, but version 0.1.12 is currently installed
train: Scanning '/content/data/orand-car-with-bbs/training/train' images and labels...5736 found, 119 missing, 0 empty, 0 corrupted: 100% 5855/5855 [00:02<00:00, 2745.39it/s]
train: New cache created: /content/data/orand-car-with-bbs/training/train.cache
val: Scanning '/content/data/orand-car-with-bbs/training/test' images and labels...634 found, 17 missing, 0 empty, 0 corrupted: 100% 651/651 [00:00<00:00, 1365.80it/s]
val: New cache created: /content/data/orand-car-with-bbs/training/test.cache
Plotting labels to runs/train/exp/labels.jpg... 

AutoAnchor: 5.66 anchors/target, 1.000 Best Possible Recall (BPR). Current anchors are a good fit to dataset ✅
Image sizes 640 train, 640 val
Using 4 dataloader workers
Logging results to runs/train/exp
Starting training for 10 epochs...

     Epoch   gpu_mem       box       obj       cls    labels  img_size
       0/9      3.5G   0.06182   0.06321   0.05458       127       640: 100% 366/366 [02:18<00:00,  2.64it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 21/21 [00:04<00:00,  4.78it/s]
                 all        651       3091       0.23      0.401       0.27      0.119

     Epoch   gpu_mem       box       obj       cls    labels  img_size
       1/9      3.5G   0.04422   0.04922   0.03813       161       640: 100% 366/366 [02:11<00:00,  2.77it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 21/21 [00:04<00:00,  5.10it/s]
                 all        651       3091      0.576      0.627      0.637      0.345

     Epoch   gpu_mem       box       obj       cls    labels  img_size
       2/9      3.5G   0.04088   0.04833   0.02547       149       640: 100% 366/366 [02:10<00:00,  2.81it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 21/21 [00:03<00:00,  5.34it/s]
                 all        651       3091      0.786      0.894      0.907      0.471

     Epoch   gpu_mem       box       obj       cls    labels  img_size
       3/9      3.5G   0.03439    0.0465   0.01476       133       640: 100% 366/366 [02:09<00:00,  2.82it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 21/21 [00:03<00:00,  5.30it/s]
                 all        651       3091      0.946       0.93      0.972      0.608

     Epoch   gpu_mem       box       obj       cls    labels  img_size
       4/9      3.5G   0.03099    0.0445     0.011       169       640: 100% 366/366 [02:09<00:00,  2.82it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 21/21 [00:04<00:00,  5.09it/s]
                 all        651       3091       0.96      0.951       0.98      0.678

     Epoch   gpu_mem       box       obj       cls    labels  img_size
       5/9      3.5G    0.0278   0.04344  0.009418       144       640: 100% 366/366 [02:10<00:00,  2.81it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 21/21 [00:03<00:00,  5.27it/s]
                 all        651       3091      0.965      0.964      0.984      0.694

     Epoch   gpu_mem       box       obj       cls    labels  img_size
       6/9      3.5G   0.02556   0.04241  0.008221       137       640: 100% 366/366 [02:10<00:00,  2.81it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 21/21 [00:03<00:00,  5.30it/s]
                 all        651       3091      0.967      0.964      0.986      0.706

     Epoch   gpu_mem       box       obj       cls    labels  img_size
       7/9      3.5G   0.02445   0.04129  0.007421       128       640: 100% 366/366 [02:10<00:00,  2.82it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 21/21 [00:03<00:00,  5.39it/s]
                 all        651       3091      0.976      0.969      0.987      0.718

     Epoch   gpu_mem       box       obj       cls    labels  img_size
       8/9      3.5G    0.0232   0.04033  0.006579       160       640: 100% 366/366 [02:10<00:00,  2.81it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 21/21 [00:03<00:00,  5.36it/s]
                 all        651       3091      0.972       0.97      0.987      0.723

     Epoch   gpu_mem       box       obj       cls    labels  img_size
       9/9      3.5G   0.02269   0.04038  0.006077       148       640: 100% 366/366 [02:09<00:00,  2.82it/s]
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 21/21 [00:03<00:00,  5.37it/s]
                 all        651       3091      0.974      0.978      0.989      0.728

10 epochs completed in 0.377 hours.
Optimizer stripped from runs/train/exp/weights/last.pt, 14.5MB
Optimizer stripped from runs/train/exp/weights/best.pt, 14.5MB

Validating runs/train/exp/weights/best.pt...
Fusing layers... 
Model Summary: 213 layers, 7037095 parameters, 0 gradients, 15.9 GFLOPs
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100% 21/21 [00:06<00:00,  3.47it/s]
                 all        651       3091      0.974      0.978      0.989      0.728
                   0        651        900      0.968      0.988       0.99      0.733
                   1        651        346      0.971      0.956      0.985      0.654
                   2        651        311      0.968      0.968      0.983      0.736
                   3        651        260      0.988       0.99      0.993      0.757
                   4        651        256      0.967       0.98      0.983      0.712
                   5        651        279       0.96      0.956      0.979      0.722
                   6        651        225      0.978      0.985      0.994      0.737
                   7        651        192      0.979      0.979      0.994      0.727
                   8        651        165      0.986      0.982      0.994      0.762
                   9        651        157      0.978      0.994      0.995      0.741
Results saved to runs/train/exp