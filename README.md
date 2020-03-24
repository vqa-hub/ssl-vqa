# ssl-vqa

## Requirement
python 3.6.8

pytorch 1.0.1 

zarr

tdqm

spacy

## download and preprocess the data

```
cd data 
bash download.sh
python preprocess_image.py --data trainval
python create_dictionary.py --dataroot vqacp2/
python preprocess_text.py --dataroot vqacp2/ --version v2
cd ..
```

## training
```
CUDA_VISIBLE_DEVICES=0 python main.py --dataroot data/vqacp2/ --img_root data/coco/ --output saved_models_cp2/ --self_loss_weight 3
```

## evaluation
* Generte a json file of results from the test set can be produced with:
```
CUDA_VISIBLE_DEVICES=0 python test.py --dataroot data/vqacp2/ --img_root data/coco/ --checkpoint_path saved_models_cp2/best_model.pth --output saved_models_cp2/result/
```
* Compute detailed accuracy each question type:
```
python comput_score.py --input saved_models_cp2/result/XX.json --dataroot data/vqacp2/
```


