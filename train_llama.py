from transformers import TrainingArguments
from datasets import load_dataset
from trl import SFTTrainer
from unsloth import FastLanguageModel

# Load our dataset from huggingface
dataset_name = "jitx/Methods2Test_java_unit_test_code"

# The dataset itself is already split into "train" and "test" splits
full_train_dataset = load_dataset(dataset_name, split="train")
full_test_dataset  = load_dataset(dataset_name, split="test")

reduce_factor = 100
reduced_train_dataset = full_train_dataset.shuffle(seed=42).select(range(len(full_train_dataset)//reduce_factor))
reduced_test_dataset = full_test_dataset.shuffle(seed=42).select(range(len(full_test_dataset)//reduce_factor))
remove_columns = ['src_fm_fc', 'src_fm_fc_co', 'src_fm_fc_ms', 'src_fm_fc_ms_ff']

max_seq_length = 2048
dtype = None
load_in_4bit = True

# The model that you want to train from the Hugging Face hub
model_name = "unsloth/llama-3-8b-bnb-4bit"

# Fine-tuned model name
new_model = "llama-3"

save_dir = '/root/model-v3' + new_model

################################################################################
# TrainingArguments parameters
################################################################################

# Output directory where the model predictions and checkpoints will be stored
output_dir = "/root/model-v3"

# Number of training epochs
num_train_epochs = 30

# Enable fp16/bf16 training (set bf16 to True with an A100)
fp16 = False
bf16 = True

# Batch size per GPU for training
per_device_train_batch_size = 4

# Batch size per GPU for evaluation
per_device_eval_batch_size = 1

# Number of update steps to accumulate the gradients for
gradient_accumulation_steps = 4

# Enable gradient checkpointing
gradient_checkpointing = True

# Maximum gradient normal (gradient clipping)
max_grad_norm = 0.3

# Initial learning rate (AdamW optimizer)
learning_rate = 2e-4

# Weight decay to apply to all layers except bias/LayerNorm weights
weight_decay = 0.01

# Optimizer to use
optim = "adamw_8bit"

# Learning rate schedule
lr_scheduler_type = "cosine"

# Ratio of steps for a linear warmup (from 0 to learning rate)
warmup_ratio = 0.03

# Group sequences into batches with same length
# Saves memory and speeds up training considerably
group_by_length = True

# Save checkpoint every X updates steps
save_steps = 0

warmup_steps = 5

training_arguments = TrainingArguments(
        per_device_train_batch_size = per_device_train_batch_size,
        gradient_accumulation_steps = gradient_accumulation_steps,
        gradient_checkpointing = gradient_checkpointing,
        warmup_steps = warmup_steps,
        num_train_epochs = num_train_epochs,
        learning_rate = learning_rate,
        fp16 = fp16,
        bf16 = bf16,
        logging_steps = 1,
        optim = optim,
        weight_decay = weight_decay,
        lr_scheduler_type = lr_scheduler_type,
        seed = 3407,
        output_dir = output_dir,
    )

# Load tokenizer and model with QLoRA configuration
# unsloth/llama-3-8b-bnb-4bit

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = model_name,
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
)

model = FastLanguageModel.get_peft_model(
    model,
    r = 64, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 16,
    lora_dropout = 0, # Supports any, but = 0 is optimized
    bias = "none",    # Supports any, but = "none" is optimized
    # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
    use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
    random_state = 3407,
    use_rslora = False,  # We support rank stabilized LoRA
    loftq_config = None, # And LoftQ
)

prompt = """
### src_fm:
{}
### src_fm_fc_ms_ff:
{}
### target:
{}
"""
# tokenizer.pad_token = tokenizer.eos_token
EOS_TOKEN = tokenizer.eos_token
def formatting_prompts_func(examples):
    fm_inputs = examples["src_fm"]
    ff_inputs = examples["src_fm_fc_ms_ff"]
    outputs = examples["target"]
    texts = []
    for fm_input, ff_input, output in zip(fm_inputs, ff_inputs, outputs):
        # Must add EOS_TOKEN, otherwise your generation will go on forever!
        text = prompt.format(fm_input, ff_input, output) + EOS_TOKEN
        texts.append(text)
    return { "text" : texts, }
pass

train_dataset = reduced_train_dataset.map(formatting_prompts_func, batched = True, remove_columns=remove_columns)
test_dataset = reduced_test_dataset.map(formatting_prompts_func, batched = True, remove_columns=remove_columns)

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = train_dataset,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    dataset_num_proc = 2,
    packing = False, # Can make training 5x faster for short sequences.
    args = training_arguments
)

trainer_stats = trainer.train()

model.save_pretrained(save_dir)