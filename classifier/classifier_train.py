import numpy as np
import pandas as pd
import torch
import shutil
from transformers import BertTokenizer, BertModel
from deeppavlov.core.common.file import read_json
from deeppavlov.dataset_readers.dstc2_reader import DSTC2DatasetReader
import csv

configs = read_json("classifier_configs.json")

DATA_PATH = configs["DATA_PATH"]
TARGET_LIST = configs["TARGET_LIST"]
CHECKPOINT_PATH = configs["CHECKPOINT_PATH"]
BEST_PATH = configs["BEST_PATH"]

val_targets=[]
val_outputs=[]

# Hyperparameters
MAX_LEN = configs["MAX_LEN"]
TRAIN_BATCH_SIZE = configs["TRAIN_BATCH_SIZE"]
VALID_BATCH_SIZE = configs["VALID_BATCH_SIZE"]
EPOCHS = configs["EPOCHS"]
LEARNING_RATE = configs["LEARNING_RATE"]

# BERT MODEL
BERT_MODEL = configs["BERT_PATH"]

TASK_ORIENTED_DATA = configs["TASK_DATA_PATH"]
FAQ_DATA_PATH =  configs["QA_DATA_PATH"]
OPEN_DOMAIN_DATA = configs["OPEN_DATA_PATH"]

def collect_data():

    task_train = DSTC2DatasetReader()._read_from_file(f"{TASK_ORIENTED_DATA}dstc2-trn.jsonlist")
    task_test = DSTC2DatasetReader()._read_from_file(f"{TASK_ORIENTED_DATA}dstc2-tst.jsonlist")
    task_valid = DSTC2DatasetReader()._read_from_file(f"{TASK_ORIENTED_DATA}dstc2-val.jsonlist")

    dataset = open(DATA_PATH, 'w', encoding="UTF-8", newline="")
    writer = csv.writer(dataset)

    writer.writerow(["text"].extend(TARGET_LIST))

    for dialog in task_train:
        text = dialog[0]["text"]
        if text:
            writer.writerow([text,0,1,0])
            writer.writerow([text,0,1,0])

    for dialog in task_test:
        text = dialog[0]["text"]
        if text:
            writer.writerow([text,0,1,0])
            writer.writerow([text,0,1,0])

    for dialog in task_valid:
        text = dialog[0]["text"]
        if text:
            writer.writerow([text,0,1,0])
            writer.writerow([text,0,1,0])

    dataset.close()

    cpt = 0
    while cpt < 768:
        with open(FAQ_DATA_PATH, 'r', encoding="UTF-8") as inp, open(DATA_PATH, 'a', encoding="UTF-8", newline="") as out:
            writer = csv.writer(out)
            for row in csv.reader(inp):
                if len(row) != 0:
                    text = row[0]
                    writer.writerow([text,1,0,0])
                    cpt += 1

    with open(OPEN_DOMAIN_DATA, 'r', encoding="UTF-8") as inp, open(DATA_PATH, 'a', encoding="UTF-8", newline="") as out:
        writer = csv.writer(out)
        for row in csv.reader(inp):
            if len(row) != 0:
                text = row[0]
                writer.writerow([text,0,0,1])

    df = pd.read_csv(DATA_PATH)
    ds = df.sample(frac=1)
    ds.to_csv(DATA_PATH)

def load_ckp(checkpoint_fpath, model, optimizer):
    """
    checkpoint_path: path to save checkpoint
    model: model that we want to load checkpoint parameters into       
    optimizer: optimizer we defined in previous training
    """
    # load check point
    checkpoint = torch.load(checkpoint_fpath)
    # initialize state_dict from checkpoint to model
    model.load_state_dict(checkpoint['state_dict'])
    # initialize optimizer from checkpoint to optimizer
    optimizer.load_state_dict(checkpoint['optimizer'])
    # initialize valid_loss_min from checkpoint to valid_loss_min
    valid_loss_min = checkpoint['valid_loss_min']
    # return model, optimizer, epoch value, min validation loss 
    return model, optimizer, checkpoint['epoch'], valid_loss_min.item()

def save_ckp(state, is_best, checkpoint_path, best_model_path):
    """
    state: checkpoint we want to save
    is_best: is this the best checkpoint; min validation loss
    checkpoint_path: path to save checkpoint
    best_model_path: path to save best model
    """
    f_path = checkpoint_path
    # save checkpoint data to the path given, checkpoint_path
    torch.save(state, f_path)
    # if it is a best model, min validation loss
    if is_best:
        best_fpath = best_model_path
        # copy that checkpoint file to best path given, best_model_path
        shutil.copyfile(f_path, best_fpath)

def loss_fn(outputs, targets):
    return torch.nn.BCEWithLogitsLoss()(outputs, targets)

def train_model(n_epochs, training_loader, validation_loader, model, 
                optimizer, checkpoint_path, best_model_path):
   
  # initialize tracker for minimum validation loss
  valid_loss_min = np.Inf
 
  for epoch in range(1, n_epochs+1):
    train_loss = 0
    valid_loss = 0

    model.train()
    print('############# Epoch {}: Training Start   #############'.format(epoch))
    print(f"TOTAL NUMBER OF BATCHES = {len(training_loader)}")
    for batch_idx, data in enumerate(training_loader):
        print(f'Step {batch_idx} / {len(training_loader)}')
        ids = data['input_ids'].to(device, dtype = torch.long)
        mask = data['attention_mask'].to(device, dtype = torch.long)
        token_type_ids = data['token_type_ids'].to(device, dtype = torch.long)
        targets = data['targets'].to(device, dtype = torch.float)

        outputs = model(ids, mask, token_type_ids)

        optimizer.zero_grad()
        loss = loss_fn(outputs, targets)
        if batch_idx%5000==0:
          print(f'Epoch: {epoch}, Training Loss:  {loss.item()}')
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        #print('before loss data in training', loss.item(), train_loss)
        train_loss = train_loss + ((1 / (batch_idx + 1)) * (loss.item() - train_loss))
        #print('after loss data in training', loss.item(), train_loss)
    
    print('############# Epoch {}: Training End     #############'.format(epoch))
    
    print('############# Epoch {}: Validation Start   #############'.format(epoch))
    ######################    
    # validate the model #
    ######################
 
    model.eval()
   
    with torch.no_grad():
      for batch_idx, data in enumerate(validation_loader, 0):
            ids = data['input_ids'].to(device, dtype = torch.long)
            mask = data['attention_mask'].to(device, dtype = torch.long)
            token_type_ids = data['token_type_ids'].to(device, dtype = torch.long)
            targets = data['targets'].to(device, dtype = torch.float)
            outputs = model(ids, mask, token_type_ids)

            loss = loss_fn(outputs, targets)
            valid_loss = valid_loss + ((1 / (batch_idx + 1)) * (loss.item() - valid_loss))
            val_targets.extend(targets.cpu().detach().numpy().tolist())
            val_outputs.extend(torch.sigmoid(outputs).cpu().detach().numpy().tolist())

      print('############# Epoch {}: Validation End     #############'.format(epoch))
      # calculate average losses
      #print('before cal avg train loss', train_loss)
      train_loss = train_loss/len(training_loader)
      valid_loss = valid_loss/len(validation_loader)
      # print training/validation statistics 
      print('Epoch: {} \tAvgerage Training Loss: {:.6f} \tAverage Validation Loss: {:.6f}'.format(
            epoch, 
            train_loss,
            valid_loss
            ))
      
      # create checkpoint variable and add important data
      checkpoint = {
            'epoch': epoch + 1,
            'valid_loss_min': valid_loss,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict()
      }
        
        # save checkpoint
      save_ckp(checkpoint, False, checkpoint_path, best_model_path)
        
      ## TODO: save the model if validation loss has decreased
      if valid_loss <= valid_loss_min:
        print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(valid_loss_min,valid_loss))
        # save checkpoint as best model
        save_ckp(checkpoint, True, checkpoint_path, best_model_path)
        valid_loss_min = valid_loss

    print('############# Epoch {}  Done   #############\n'.format(epoch))

  return model

class CustomDataset(torch.utils.data.Dataset):

    def __init__(self, df, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.df = df
        self.title = df['text']
        self.targets = self.df[TARGET_LIST].values
        self.max_len = max_len

    def __len__(self):
        return len(self.title)

    def __getitem__(self, index):
        title = str(self.title[index])
        title = " ".join(title.split())

        inputs = self.tokenizer.encode_plus(
            title,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            return_token_type_ids=True,
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        return {
            'input_ids': inputs['input_ids'].flatten(),
            'attention_mask': inputs['attention_mask'].flatten(),
            'token_type_ids': inputs["token_type_ids"].flatten(),
            'targets': torch.FloatTensor(self.targets[index])
        }

class BERTClass(torch.nn.Module):
    def __init__(self):
        super(BERTClass, self).__init__()
        self.bert_model = BertModel.from_pretrained(BERT_MODEL, return_dict=True)
        self.dropout = torch.nn.Dropout(0.3)
        self.linear = torch.nn.Linear(768, 3)
    
    def forward(self, input_ids, attn_mask, token_type_ids):
        output = self.bert_model(
            input_ids, 
            attention_mask=attn_mask, 
            token_type_ids=token_type_ids
        )
        output_dropout = self.dropout(output.pooler_output)
        output = self.linear(output_dropout)
        return output

collect_data()

df = pd.read_csv(DATA_PATH)
train_size = 0.8
train_df = df.sample(frac=train_size, random_state=200).reset_index(drop=True)
val_df = df.drop(train_df.index).reset_index(drop=True)

tokenizer = BertTokenizer.from_pretrained(BERT_MODEL)
train_dataset = CustomDataset(train_df, tokenizer, MAX_LEN)
valid_dataset = CustomDataset(val_df, tokenizer, MAX_LEN)

train_data_loader = torch.utils.data.DataLoader(train_dataset, 
    batch_size=TRAIN_BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

val_data_loader = torch.utils.data.DataLoader(valid_dataset, 
    batch_size=VALID_BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

model = BERTClass()
model.to(device)

optimizer = torch.optim.Adam(params =  model.parameters(), lr=LEARNING_RATE)

trained_model = train_model(EPOCHS, train_data_loader, val_data_loader, model, optimizer, CHECKPOINT_PATH, BEST_PATH)