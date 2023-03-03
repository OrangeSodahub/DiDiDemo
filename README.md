# DiDiDemo

### Build dataset
```bash
python dataset/create_dataset.py
```

### Evaluation on CCPD dataset
recognition:
```bash
python tools/eval.py -c ../models/rec_ppocr_v3_finetune/config.yml -o \
    Global.pretrained_model=../models/rec_ppocr_v3_finetune/best_accuracy.pdparams \
    Eval.dataset.data_dir=/home/zonlin/PaddlePaddle/DiDiDemo/data/CCPD_processed \
    Eval.dataset.label_file_list=[/home/zonlin/PaddlePaddle/DiDiDemo/data/CCPD_processed/test/rec.txt]
```

### End-to-end inference
```bash
python tools/infer/predict_system.py \
    --det_model_dir=output/CCPD/det/infer/ \
    --rec_model_dir=output/CCPD/rec/infer/ \
    --image_dir="/home/aistudio/data/CCPD2020/ccpd_green/test/04131106321839081-92_258-159&509_530&611-527&611_172&599_159&509_530&525-0_0_3_32_30_31_30_30-109-106.jpg" \
    --rec_image_shape=3,48,320
```

### Demo
```bash
python demo.py
```

### Evalution on CCPD dataset [[logs]](https://github.com/OrangeSodahub/DiDiDemo/blob/master/backup_test.log):
```
total 5006 precision 0.7843445008846527 precision_ 0.8725396825396825
fps 12.30353074789225 count_list [193, 280, 73, 39, 41, 64, 212, 3598, 506]
```
