import os
from datasets import load_dataset 
from transformers import AutoTokenizer
from transformers import EncoderDecoderModel
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer
from deeppavlov.core.common.file import read_json
from sacrebleu import corpus_bleu

configs = read_json("open_configs.json")

encoder_max_length = configs["ENCODER_LENGTH"]
decoder_max_length = configs["DECODER_LENGTH"]
MODEL_NAME = configs["BERT_MODEL"]
SAVE_PATH = configs["SAVE_PATH"]

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

all_data = load_dataset("dataset.py")
train_data = all_data['train'].train_test_split(test_size=0.1,seed=42)['train']
val_data = all_data['train'].train_test_split(test_size=0.1,seed=42)['test']
dev_data = val_data.train_test_split(test_size=0.5,seed=42)['train']
test_data = val_data.train_test_split(test_size=0.5,seed=42)['test']


#Lengths of train/dev/test sets
print("Length of train data",len(train_data))
print("Length of dev data",len(dev_data))
print("Length of test data",len(test_data))

def process_data_to_model_inputs(batch):                                                               
    # Tokenizer will automatically set [BOS] <text> [EOS]                                               
    inputs = tokenizer(batch["context"], padding="max_length", truncation=True, max_length=encoder_max_length)
    outputs = tokenizer(batch["response"], padding="max_length", truncation=True, max_length=decoder_max_length)
                                                                                                        
    batch["input_ids"] = inputs.input_ids                                                               
    batch["attention_mask"] = inputs.attention_mask                                                     
    batch["decoder_input_ids"] = outputs.input_ids                                                      
    batch["labels"] = outputs.input_ids.copy()                                                          
    # mask loss for padding                                                                             
    batch["labels"] = [                                                                                 
        [-100 if token == tokenizer.pad_token_id else token for token in labels] for labels in batch["labels"]
    ]                     
    batch["decoder_attention_mask"] = outputs.attention_mask                                                                              
                                                                                                         
    return batch


batch_size=8

train_data = train_data.map(
    process_data_to_model_inputs, 
    batched=True, 
    batch_size=batch_size, 
    remove_columns=["context", "response"],
)
train_data.set_format(
    type="torch", columns=["input_ids", "attention_mask", "decoder_input_ids", "decoder_attention_mask", "labels"],
)

dev_data = dev_data.map(
    process_data_to_model_inputs, 
    batched=True, 
    batch_size=batch_size, 
    remove_columns=["context", "response"],
)
dev_data.set_format(
    type="torch", columns=["input_ids", "attention_mask", "decoder_input_ids", "decoder_attention_mask", "labels"],
)

test_data = test_data.map(
    process_data_to_model_inputs, 
    batched=True, 
    batch_size=batch_size, 
    remove_columns=["context", "response"],
)
test_data.set_format(
    type="torch", columns=["input_ids", "attention_mask", "decoder_input_ids", "decoder_attention_mask", "labels"],
)


bert2bert = EncoderDecoderModel.from_encoder_decoder_pretrained(MODEL_NAME, MODEL_NAME, tie_encoder_decoder=False)

#set special tokens
bert2bert.config.decoder_start_token_id = tokenizer.cls_token_id                                             
bert2bert.config.eos_token_id = tokenizer.sep_token_id
bert2bert.config.pad_token_id = tokenizer.pad_token_id

#sensible parameters for beam search
#set decoding params                               
bert2bert.config.max_length = 64
bert2bert.config.early_stopping = True

bert2bert.config.num_beams = 1
bert2bert.config.vocab_size = bert2bert.config.encoder.vocab_size

def compute_metrics(pred):
  labels_ids = pred.label_ids
  #pred_ids = torch.argmax(pred.predictions,dim=2)
  pred_ids = pred.predictions  

  # all unnecessary tokens are removed
  pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
  labels_ids[labels_ids == -100] = tokenizer.pad_token_id
  label_str = tokenizer.batch_decode(labels_ids, skip_special_tokens=True)

  return {"bleu": round(corpus_bleu(pred_str , [label_str]).score, 4)}

  #Set training arguments 
training_args = Seq2SeqTrainingArguments(
    output_dir=".\model",
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size//2,
    gradient_accumulation_steps = 2,
    predict_with_generate=True,
    do_eval=True,
    evaluation_strategy ="epoch",
    do_train=True,
    logging_steps=500,  
    save_steps= 32965 // ( batch_size * 2),  
    warmup_steps=100,
    eval_steps=10,
    # max_steps=1000, # delete for full training
    num_train_epochs=3,# uncomment for full training
    overwrite_output_dir=True,
    save_total_limit=0,
    #fp16=True, 
)

trainer = Seq2SeqTrainer(
    model=bert2bert,
    args=training_args,
    compute_metrics=compute_metrics,
    train_dataset=train_data,
    eval_dataset=dev_data,
    tokenizer=tokenizer
)

trainer.train()
trainer.evaluate()

trainer._save(SAVE_PATH)
tokenizer.save_pretrained(SAVE_PATH)