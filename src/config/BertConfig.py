
mode = 'image_caption' # caption|image|image_caption|
model = 'nghuyong/ernie-2.0-en'

do_lower = False
token_type_ids = False

#model = 'distilbert-base-uncased'
#model = 'roberta-base'
#model = 'nghuyong/ernie-2.0-en'

hidden_size = 768

#Data
img_col = 'g5'
img_sep = ','
separator = ' . '
train_batch_size = 16 # Recomended 16 or 32
dev_batch_size = 16
max_len_tokenized_sentence = -1 # -1 to calculate 

#Training
epochs =  5

optimizer = 'AdamW'
lr = 2e-5 # Recomended 5e-5 - 2e-5
eps = 1e-8 # Recomended
weight_decay = 0.01

scheduler = 'None' #linear
warmup_steps = int(8000/train_batch_size)

swa_start = 0
swa_scheduler = 'linear' # linear or cosine

gradient_accumulation_steps = 1
clipping_grad_norm = 2
