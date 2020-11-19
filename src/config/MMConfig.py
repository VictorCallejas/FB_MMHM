
mode = 'image_caption' # caption|image|image_caption|

max_seq_length = 170
model ='uniter-base' # lxmert, uniter-base, visualbert-base, uniter-large
pretrained_weights = '../artifacts/MM/uniter-base.pt'

frcnn_path = '../data/BUTD_features/'

hidden_size = 768

#Data
img_col = 'g5'
img_sep = ','
separator = ' . '
train_batch_size = 32 # Recomended 16 or 32
dev_batch_size = 32
#max_len_tokenized_sentence = -1 # -1 to calculate 

#Training
epochs =  12 # Recomended 8 - 12
optimizer = 'AdamW'
lr = 2e-5 # Recomended 5e-5 
eps = 1e-8 # Recomended
weight_decay = 0.01

scheduler = 'None' #linear
warmup_steps = int(8000/train_batch_size)

swa_start = 5
swa_scheduler = 'linear' # linear or cosine

gradient_accumulation_steps = 8
clipping_grad_norm = 2

