<<<<<<< HEAD
<h1 align="center">
  Open-Chinese-Llama2-Chat
</h1>
<p align="center">
  <font face="黑体" color=orange size="6"> Llama-2中文预训练大模型 </font>
</p>

## 简介
  本项目基于Llama-2,面向不同的业务场景下的垂直预料, 实现了包括二次预训练、有监督微调、奖励建模、强化学习训练。


- [训练数据](#-数据来源)
- [模型训练](#-模型训练)
  - [扩充词表](#-扩充词表)
  - [二次预训练](#-预训练模型)
    - [stage_one](#-第一阶段预训练)
    - [stage_two](#-第二阶段预训练)
  - [模型微调](#-模型微调)
    - [微调脚本](#微调脚本)
    - [数据格式](#数据格式)
  - [奖励模型](#-奖励模型)
    - [微调脚本](#微调脚本)
    - [数据格式](#数据格式)
  - [RLHF强化学习](#-模型微调)
- [模型推理](#-模型推理)
- [🚀 推理加速](#-推理加速)
  - [vLLM](#vllm)
- [相关资料](#-学习资料)
  - [Llama相关论文](#llama相关论文)
- [问题反馈](#-问题反馈)



## 数据来源

我们计划通过以下数据来优化Llama2的中文能力:

| 类型                                                       | 描述                                                         |
| ---------------------------------------------------------- | ------------------------------------------------------------ |
| [Wikipedia](https://github.com/goldsmith/Wikipedia)        | 中文Wikipedia的数据                                          |
| [BelleGroup](https://huggingface.co/datasets/BelleGroup/train_1M_CN)                | 约100万条由BELLE项目生成的中文指令数据。                                       |
| [ShareGPT-CN](https://huggingface.co/datasets/FreedomIntelligence/ShareGPT-CN) | ShareGPT中文版   |


### 预训练模型

Llama2预训练模型包含7B、13B和70B三个版本

| 模型名称   | 模型加载名称             | 下载地址                                                     |
| ---------- | ------------------------- | ------------------------------------------------------------ |
| Llama2-7B  | meta-llama/Llama-2-7b-hf  | [模型下载](https://huggingface.co/meta-llama/Llama-2-7b-hf)  |
| Llama2-13B | meta-llama/Llama-2-13b-hf | [模型下载](https://huggingface.co/meta-llama/Llama-2-13b-hf) |
| Llama2-70B | meta-llama/Llama-2-70b-hf | [模型下载](https://huggingface.co/meta-llama/Llama-2-70b-hf) |

### Chat模型

Llama2-Chat模型基于预训练模型进行了监督微调，具备更强的对话能力

| 模型名称        | 模型加载名称                  | 下载地址                                                     |
| --------------- | ------------------------------ | ------------------------------------------------------------ |
| Llama2-7B-Chat  | meta-llama/Llama-2-7b-chat-hf  | [模型下载](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf) |
| Llama2-13B-Chat | meta-llama/Llama-2-13b-chat-hf | [模型下载](https://huggingface.co/meta-llama/Llama-2-13b-chat-hf) |
| Llama2-70B-Chat | meta-llama/Llama-2-70b-chat-hf | [模型下载](https://huggingface.co/meta-llama/Llama-2-70b-chat-hf) |


# 模型训练

## 扩充词表
由于原版LLaMA2和Llama的词表都只有32000, 对中文的支持不太友好，所以本项目在LLaMA2的基础上进一步扩充了中文词表。
参考了Alpaca项目，针对Llama-2模型新增了新版中文词表并与原版LLaMA模型的32K词表进行合并
排除重复的token后，得到的最终中文LLaMA词表大小为55296。
- 提供代码merge_tokenizers.py参考参考代码扩充词表。该脚本运行方式如下：

```bash
  python merge_tokenizers.py \
  --llama_tokenizer_dir ./tokenizer_model/ \
  --chinese_sp_model_file ./chinese_sp.model
```
- llama_tokenizer_dir：原llama词表路径目录
- chinese_sp.model：新词表文件

## 预训练模型
预训练本项目分为两个阶段，第一阶段仅训练embedding层，目的是让扩充的中文词表更好的适应模型。但是弊端是该阶段收敛速递会非常慢，
如果不是有特别充裕的时间和计算资源，也可以跳过该阶段，直接进行第二阶段的预训练。

- ##第一阶段预训练
    第一阶段的预训练代码，本项目是唯一开源出来的，运行如下命令供参考：
    ```bash
    CUDA_VISIBLE_DEVICES=2,3 torchrun --master_port 29510 --nproc_per_node=2 run_clm.py \
        --model_name_or_path ./Llama-2-13b-hf/ \
        --train_file ./wiki_embeddings_data.txt \
        --per_device_train_batch_size 1 \
        --per_device_eval_batch_size 1 \
        --do_train \
        --do_eval False \
        --output_dir ./Llama-2-13b-embeddings_v1/ \
        --gradient_accumulation_steps 16 \
        --gradient_checkpointing True \
        --block_size 4096 \
        --logging_steps 2 \
        --save_steps=500 \
        --num_train_epochs 1
    ```
  这一阶段的的数据集示例如下：

- ##第二阶段预训练
   第二阶段预训练使用LoRA技术(论文“[LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)”以及源码[LoRA](https://github.com/microsoft/LoRA))，训练embedding的同时也更新LoRA参数。
   脚本运行如下命令供参考：
   ```bash
    CUDA_VISIBLE_DEVICES=2,3 torchrun --master_port 29510 --nproc_per_node=2 run_clm_pt_with_peft.py \
    --deepspeed ds_zero2_no_offload.json \
    --model_name_or_path /data_hxs/Llama-2-13b-embeddings_v1/ \
    --tokenizer_name_or_path /data_hxs/Llama-2-13b-embeddings_v1/ \
    --dataset_dir /data_hxs/data_pt/ \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --do_train \
    --fp16 \
    --num_train_epochs 3 \
    --lr_scheduler_type cosine \
    --learning_rate 2e-4 \
    --warmup_ratio 0.05 \
    --weight_decay 0.01 \
    --logging_strategy steps \
    --logging_steps 10 \
    --save_strategy steps \
    --save_total_limit 3 \
    --save_steps 500 \
    --gradient_accumulation_steps 16 \
    --preprocessing_num_workers 8 \
    --block_size 4096 \
    --output_dir /data_hxs/Llama-2-13b-pt_v1/ \
    --overwrite_output_dir \
    --ddp_timeout 30000 \
    --logging_first_step True \
    --lora_rank 64 \
    --lora_alpha 128 \
    --lora_dropout 0.05 \
    --modules_to_save "embed_tokens,lm_head" \
    --torch_dtype float16 \
    --gradient_checkpointing True \
    --ddp_find_unused_parameters False
    ```

    #### 合并脚本    
    训练完成后，得到的是Lora的权重，需要进一步合并成hf格式的模型文件，参考脚步命令如下：
    
      ```bash
      python merge_llama_with_chinese_lora.py \
        --base_model /data_hxs/output/llama2-13b/ \
        --lora_model /data_hxs/output/llama2-13b-lora/ \
        --output_type huggingface \
        --output_dir /data_hxs/output/llama2-13b_merge
      ```
    这一阶段的的数据集示例如下：


## 模型微调
SFT有监督微调，构造指令微调数据集，在上一步预训练模型基础上做指令精调，目的是对齐指令意图。

  #### 微调脚本
  脚本运行如下命令供参考：
  ```bash
  CUDA_VISIBLE_DEVICES=2,3,4,5 torchrun --master_port 29510 --nproc_per_node=4 run_clm_sft_with_peft.py \
      --deepspeed ./ds_zero2_no_offload.json \
      --model_name_or_path path/to/model \
      --dataset_dir path/to/sft/data/dir \
      --validation_split_percentage 0.1 \
      --per_device_train_batch_size 2 \
      --per_device_eval_batch_size 2 \
      --do_train \
      --do_eval \
      --seed $RANDOM \
      --fp16 \
      --num_train_epochs 3 \
      --lr_scheduler_type cosine \
      --learning_rate 2e-4 \
      --warmup_ratio 0.03 \
      --weight_decay 0 \
      --logging_strategy steps \
      --logging_steps 10 \
      --save_strategy steps \
      --save_total_limit 3 \
      --evaluation_strategy steps \
      --eval_steps 250 \
      --save_steps 500 \
      --gradient_accumulation_steps 16 \
      --preprocessing_num_workers 8 \
      --max_seq_length 512 \
      --output_dir path/to/output \
      --overwrite_output_dir \
      --ddp_timeout 30000 \
      --logging_first_step True \
      --lora_rank 8 \
      --lora_alpha 32 \
      --lora_dropout 0.05 \
      --torch_dtype float16 \
      --gradient_checkpointing True \
      --ddp_find_unused_parameters False
  ```

  - 训练完成后，得到的是Lora的权重，需要进一步合并成hf格式的模型文件，同上。

  #### 数据格式
  这一阶段的的数据集示例如下, 如下格式的json文件：
  ```
  [
    {
      "instruction": "提出一种对给定数据进行分类的方法。",
      "input": "不同类型鱼的列表",
      "output": "可以根据它们的栖息地对数据进行分类，例如淡水鱼、海水鱼和河鱼。也可以根据它们的地理位置进行分类，例如生活在温带或热带地区的鱼类。此外，数据还可以根据它们食用的食物类型进行分类，例如食草鱼、杂食鱼或食肉鱼。"
    }
  ]
  ```

## 奖励模型
构造人类偏好排序数据集，训练奖励模型，用来对齐人类偏好。
  
  #### 微调脚本
  - 脚本运行如下命令供参考：
    ```bash
      CUDA_VISIBLE_DEVICES=4 python rm.py \
        --model_type llama \
        --model_name_or_path ./Llama-2-13b-hf/ \
        --tokenizer_name_or_path ./tokenizer_path/ \
        --train_file_dir ./reward \
        --validation_file_dir ./reward \
        --per_device_train_batch_size 2 \
        --per_device_eval_batch_size 2 \
        --do_train \
        --use_peft True \
        --seed 42 \
        --max_train_samples 1000 \
        --max_eval_samples 10 \
        --num_train_epochs 1 \
        --learning_rate 2e-5 \
        --warmup_ratio 0.05 \
        --weight_decay 0.001 \
        --logging_strategy steps \
        --logging_steps 1 \
        --evaluation_strategy no \
        --save_steps 500 \
        --save_strategy steps \
        --save_total_limit 3 \
        --max_source_length 256 \
        --max_target_length 256 \
        --output_dir ./Llama-2-13b-rw_v1 \
        --overwrite_output_dir \
        --ddp_timeout 30000 \
        --logging_first_step True \
        --target_modules all \
        --lora_rank 8 \
        --lora_alpha 16 \
        --lora_dropout 0.05 \
        --torch_dtype float32 \
        --device_map auto \
        --report_to tensorboard \
        --ddp_find_unused_parameters False \
        --remove_unused_columns False \
        --gradient_checkpointing True
    ```
    #### 数据格式
      这一阶段的的数据集示例如下, 如下格式的json文件：
      ```
      {"question": "孩子感统失调，学习不行怎么办？，孩子说话晚，走路晚，现在小学二年级，学习跟不上，理解力差，运动协调性差，家里很着急，不知怎么办。", "response_chosen": "病情分析：你好!孩子说话晚，走路也晚，很可能是大脑本身发育不好引起的发育迟缓。而五岁时所致的智力检查为临界范围，那就是说孩子的智商是有问题的，也应考虑与大脑发育不好有关。指导意见：人的大脑在头一年发育最快，可塑性最强，在头三年可塑性还是可以的，超过三岁再进行训练，效果就不怎么好了。建议再给孩子做一做智力测试，如果孩子的智商还是在临界范围，那就要考虑让孩子去特殊学校进行康复训练，而不是继续在普通小学就读，否则对孩子来说，就是强人所难了。希望自己的孩子能聪明，这是每个家长都会有的心愿，但如果孩子自身的条件就是不能跟上同龄孩子，那家长也要面对这个事实的，对吗？医生询问：", "response_rejected": "建议家长先带孩子去正规医院做全面检查以确定病因和病情严重程度；同时可以进行物理治疗、康复训练等辅助治疗方法。"}
      ```
## RLHF 强化学习
基于人类反馈的强化学习(RLHF)，用奖励模型来训练SFT模型，生成模型使用奖励或惩罚来更新其策略，以便生成更高质量、更符合人类偏好的文本
敬请期待，不久后将公布。


# 模型推理
- 模型调用代码示例
  如果你下载的是完整版权重，或者之前已执行了合并脚本将LoRA权重与原版Llama-2合并，可直接加载完整版模型。
  ```bash
  CUDA_VISIBLE_DEVICES=4 python inference/inference_hf.py \
      --base_model /data_hxs/code/Chinese-LLaMAA/Llama-2-13b-hf/ \
      --interactive \
      --with_prompt \
      --gpus 4
  ```

### Llama相关论文
* [LLaMA: Open and Efficient Foundation Language Models](https://arxiv.org/abs/2302.13971)
* [Llama 2: Open Foundation and Fine-Tuned Chat Models](https://arxiv.org/abs/2307.09288)

## 问题反馈
如有问题，请在Issue中提交，在提交问题之前，请先查阅以往的issue是否能解决你的问题。
=======
# open-llama2
从预训练到强化学习的中文llama2
>>>>>>> 441f52eeab95af54fef3af1b1f93a20c2547d75d
