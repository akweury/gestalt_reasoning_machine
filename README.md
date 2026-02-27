# Gestalt Reasoning Machine

### Setup Locally

1. install pytorch

```
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
```

2. install requirements.txt

``` 
pip install -r requirements.txt
```

----

### Docker
```
docker build -t grm:latest .
``` 

```
docker run -it --gpus all --rm grm:latest

``` 

#### Train: Baseline Models

##### ViT
``` 
python -m baselines.eval_models --batch_size 1 --principle proximity --img_num 3 --model vit --device_id 6 --img_size 224 --remote
python -m baselines.eval_models --batch_size 1 --principle similarity --img_num 3 --model vit --device_id 3 --img_size 224 --remote
python -m baselines.eval_models --batch_size 1 --principle closure --img_num 3 --model vit --device_id 10 --img_size 224 --remote
python -m baselines.eval_models --batch_size 1 --principle symmetry --img_num 3 --model vit --device_id 10 --img_size 224 --remote
python -m baselines.eval_models --batch_size 1 --principle continuity --img_num 3 --model vit --device_id 5 --img_size 224 --remote

```

##### Llava-7B
``` 
python -m baselines.eval_models --batch_size 1 --principle proximity --img_num 3 --model llava --device_id 2 --img_size 224 --remote
python -m baselines.eval_models --batch_size 1 --principle similarity --img_num 3 --model llava --device_id 3 --img_size 224 --remote
python -m baselines.eval_models --batch_size 1 --principle closure --img_num 3 --model llava --device_id 5 --img_size 224 --remote
python -m baselines.eval_models --batch_size 1 --principle symmetry --img_num 3 --model llava --device_id 6 --img_size 224 --remote
python -m baselines.eval_models --batch_size 1 --principle continuity --img_num 3 --model llava --device_id 0 --img_size 224 --remote
```

##### InternVL-78B
``` 
CUDA_VISIBLE_DEVICES=0,1,2 python -m baselines.eval_models --batch_size 1 --principle proximity --img_num 3 --model internVL3_78B --device_id 0 --img_size 224 --remote
CUDA_VISIBLE_DEVICES=0,1,2 python -m baselines.eval_models --batch_size 1 --principle similarity --img_num 3 --model internVL3_78B --device_id 0 --img_size 224 --remote
CUDA_VISIBLE_DEVICES=3,4,5 python -m baselines.eval_models --batch_size 1 --principle closure --img_num 3 --model internVL3_78B --device_id 0 --img_size 224 --remote
CUDA_VISIBLE_DEVICES=4,5,7 python -m baselines.eval_models --batch_size 1 --principle symmetry --img_num 3 --model internVL3_78B --device_id 0 --img_size 224 --remote
CUDA_VISIBLE_DEVICES=3,4,5 python -m baselines.eval_models --batch_size 1 --principle continuity --img_num 3 --model internVL3_78B --device_id 0 --img_size 224 --remote
```

# train gpt5
```
python -m baselines.eval_models --batch_size 1 --principle proximity --model gpt5 --img_num 3 --device_id 0 --img_size 224 --remote
python -m baselines.eval_models --batch_size 1 --principle similarity --model gpt5 --img_num 3 --device_id 0 --img_size 224 --remote
python -m baselines.eval_models --batch_size 1 --principle closure --model gpt5 --img_num 3 --device_id 0 --img_size 224 --remote
python -m baselines.eval_models --batch_size 1 --principle symmetry --model gpt5 --img_num 3 --device_id 0 --img_size 224 --remote
python -m baselines.eval_models --batch_size 1 --principle continuity --model gpt5 --img_num 3 --device_id 0 --img_size 224 --remote
```


#### Train: Ablation Study
```
python -m src.ablation_study --device 10 --task_id 6 --line_min_size 3
python -m src.ablation_study --device 0 --principle closure
python -m src.ablation_study --device 1 --principle similarity
python -m src.ablation_study --device 0 --principle proximity --remote
python -m src.ablation_study --device 5 --principle continuity --remote
python -m src.ablation_study --device 7 --principle symmetry --remote


#### Analysis Results

``` 
python -m src.analysis_results --model principle --model vit --principle proximity --img_num 3  
python -m src.analysis_results --model principle --model vit --principle similarity --img_num 3  
python -m src.analysis_results --model principle --model vit --principle closure --img_num 3  
python -m src.analysis_results --model principle --model vit --principle symmetry --img_num 3  
python -m src.analysis_results --model principle --model vit --principle continuity --img_num 3  

python -m src.analysis_results --model principle --model llava --principle proximity --img_num 3  
python -m src.analysis_results --model principle --model llava --principle similarity --img_num 3  
python -m src.analysis_results --model principle --model llava --principle closure --img_num 3  
python -m src.analysis_results --model principle --model llava --principle symmetry --img_num 3  
python -m src.analysis_results --model principle --model llava --principle continuity --img_num 3  

python -m src.analysis_results --model principle --model internVL3-78B --principle proximity --img_num 3  
python -m src.analysis_results --model principle --model internVL3-78B --principle similarity --img_num 3  
python -m src.analysis_results --model principle --model internVL3-78B --principle closure --img_num 3  
python -m src.analysis_results --model principle --model internVL3-78B --principle symmetry --img_num 3  
python -m src.analysis_results --model principle --model internVL3-78B --principle continuity --img_num 3  

python -m src.analysis_results --model principle --model GRM --principle proximity --img_num 3  
python -m src.analysis_results --model principle --model GRM --principle similarity --img_num 3  
python -m src.analysis_results --model principle --model GRM --principle closure --img_num 3  
python -m src.analysis_results --model principle --model GRM --principle symmetry --img_num 3  
python -m src.analysis_results --model principle --model GRM --principle continuity --img_num 3

```
### Grp Comparison
python -m src.elvis_exp.grp_comparison --remote --device 6


---
#### Others
```
pip install -r requirements.txt
pip uninstall flash_attn flash_attn_2
```


python -m src.metric_od_gd --principle continuity --device 5
python -m mbg.group.train_gd_transformer --remote --task_num 100 --epochs 200 --principle proximity --device 2
python -m mbg.group.train_gd_transformer --remote --task_num 100 --epochs 1000 --principle similarity --device 6
python -m src.ablation_study --device 2 --principle proximity --remote
python -m src.ablation_study --device 5 --principle similarity --remote
python -m src.ablation_study --device 4 --principle closure --remote