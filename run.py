import torch
from torchtext.datasets import AG_NEWS
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torch.utils.data import DataLoader
from torch import nn
from torch.utils.data.dataset import random_split
from torchtext.data.functional import to_map_style_dataset
import time
from configurations import configs

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = "cpu"

# Access to the raw dataset iterators
# next(train_iter)


# Prepare data processing pipelines

tokenizer = get_tokenizer('basic_english')
train_iter = AG_NEWS(split='train')

def yield_tokens(data_iter):
    for _, text in data_iter:
        yield tokenizer(text)

vocab = build_vocab_from_iterator(yield_tokens(train_iter), specials=["<unk>"])
vocab.set_default_index(vocab["<unk>"])

print(vocab(['here', 'is', 'an', 'example']))

text_pipeline = lambda x: vocab(tokenizer(x))
label_pipeline = lambda x: int(x) - 1


# Generate data batch and iterator

def collate_batch(batch):
    label_list, text_list, offsets = [], [], [0]
    for (_label, _text) in batch:
         label_list.append(label_pipeline(_label))
         processed_text = torch.tensor(text_pipeline(_text), dtype=torch.int64)
         text_list.append(processed_text)
         offsets.append(processed_text.size(0))
    label_list = torch.tensor(label_list, dtype=torch.int64)
    offsets = torch.tensor(offsets[:-1]).cumsum(dim=0)
    text_list = torch.cat(text_list)
    return label_list.to(device), text_list.to(device), offsets.to(device)

train_iter = AG_NEWS(split='train')
dataloader = DataLoader(train_iter, batch_size=8, shuffle=False, collate_fn=collate_batch)


# Define the model

class TextClassificationModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_class):
        super(TextClassificationModel, self).__init__()
        self.embedding = nn.EmbeddingBag(vocab_size, embed_dim, sparse=True)
        self.fc = nn.Linear(embed_dim, num_class)
        self.init_weights()

    def init_weights(self):
        initrange = 0.5
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.fc.weight.data.uniform_(-initrange, initrange)
        self.fc.bias.data.zero_()

    def forward(self, text, offsets):
        embedded = self.embedding(text, offsets)
        return self.fc(embedded)

# Labels

# 1 : World
# 2 : Sports
# 3 : Business
# 4 : Sci/Tec

# Params
num_class = len(set([label for (label, text) in train_iter]))
vocab_size = len(vocab)
emsize = 64
model = TextClassificationModel(vocab_size, emsize, num_class).to(device)
print(model)


# Define functions to train the model and evaluate results

def train(dataloader, optimizer, criterion, epoch, verbose):
    model.train()
    total_acc, total_count = 0, 0
    log_interval = 500
    start_time = time.time()

    for idx, (label, text, offsets) in enumerate(dataloader):
        optimizer.zero_grad()
        predicted_label = model(text, offsets)
        loss = criterion(predicted_label, label)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
        optimizer.step()
        total_acc += (predicted_label.argmax(1) == label).sum().item()
        total_count += label.size(0)
        if idx % log_interval == 0 and idx > 0 and verbose:
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches '
                  '| accuracy {:8.3f}'.format(epoch, idx, len(dataloader),
                                              total_acc/total_count))
            total_acc, total_count = 0, 0
            start_time = time.time()

def evaluate(dataloader, criterion):
    model.eval()
    total_acc, total_count = 0, 0

    with torch.no_grad():
        for idx, (label, text, offsets) in enumerate(dataloader):
            predicted_label = model(text, offsets)
            loss = criterion(predicted_label, label)
            total_acc += (predicted_label.argmax(1) == label).sum().item()
            total_count += label.size(0)
    return total_acc/total_count


# Split the dataset and run the model

def start_train(batch_size, epochs, lr, verbose):
    BATCH_SIZE=batch_size
    EPOCHS=epochs
    LR=lr
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.1)
    total_accu = None
    train_iter, test_iter = AG_NEWS()
    train_dataset = to_map_style_dataset(train_iter)
    test_dataset = to_map_style_dataset(test_iter)
    num_train = int(len(train_dataset) * 0.95)
    split_train_, split_valid_ = \
        random_split(train_dataset, [num_train, len(train_dataset) - num_train])

    train_dataloader = DataLoader(split_train_, batch_size=BATCH_SIZE,
                                shuffle=True, collate_fn=collate_batch)
    valid_dataloader = DataLoader(split_valid_, batch_size=BATCH_SIZE,
                                shuffle=True, collate_fn=collate_batch)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE,
                                shuffle=True, collate_fn=collate_batch)
    
    train_start_time = time.time()
    for epoch in range(1, EPOCHS + 1):
        epoch_start_time = time.time()
        train(train_dataloader, optimizer, criterion, epoch, verbose)
        accu_val = evaluate(valid_dataloader, criterion)
        if total_accu is not None and total_accu > accu_val:
            scheduler.step()
        else:
            total_accu = accu_val
        
        print('-' * 59)
        print('| end of epoch {:3d} | time: {:5.2f}s | '
            'valid accuracy {:8.3f} '.format(epoch,
                                            time.time() - epoch_start_time,
                                            accu_val))
        print('-' * 59)

    print("Whole train time: {:5.2f}s".format(time.time() - train_start_time))

    print('Checking the results of test dataset.')
    accu_test = evaluate(test_dataloader, criterion)
    print('test accuracy {:8.3f}'.format(accu_test))


## GPU: GTX 1660 Ti          => Whole train time: 41.31s
## CPU: i9 9900K 8 threads   => Whole train time: 63.44s
## CPU: i9 9900K 16 threads  => Whole train time: 84.02s


for conf in configs:
    batch_size = conf["BATCH_SIZE"]
    epochs = conf["EPOCHS"]
    lr = conf["LR"]
    verbose = conf["VERBOSE"]
    print('')
    print('')
    print('*' * 59)
    print(conf)
    print('')
    start_train(batch_size, epochs, lr, verbose)
    print('*' * 59)