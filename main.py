#! /usr/bin/env python
import dill
import os
import argparse
import datetime
import torch
import torchtext.data as data
from torchtext.vocab import Vectors
import torchtext.datasets as datasets
import model
import train
import mydatasets
import load_data
from selfAttention import SelfAttention


parser = argparse.ArgumentParser(description='CNN text classificer')
parser.add_argument('-data-path', type=str, default="data", help='path of data which contains train.txt , dev.txt ,test.txt')
# learning

parser.add_argument('-lr', type=float, default=0.0002, help='initial learning rate [default: 0.0002]')
parser.add_argument('-epochs', type=int, default=100, help='number of epochs for train [default: 100]')
parser.add_argument('-batch-size', type=int, default=32, help='batch size for training [default: 64]')
parser.add_argument('-log-interval',  type=int, default=100,   help='how many steps to wait before logging training status [default: 1]')
parser.add_argument('-test-interval', type=int, default=500, help='how many steps to wait before testing [default: 100]')
parser.add_argument('-save-interval', type=int, default=500, help='how many steps to wait before saving [default:500]')
parser.add_argument('-save-dir', type=str, default='snapshot', help='where to save the snapshot')
parser.add_argument('-early-stop', type=int, default=10000, help='iteration numbers to stop without performance increasing')
parser.add_argument('-save-best', type=bool, default=True, help='whether to save when get best performance')
# data 
parser.add_argument('-shuffle', action='store_true', default=True, help='shuffle the data every epoch')
parser.add_argument('-max-length', type=int, default=600, help='sentence max length')
# model
parser.add_argument('-dropout', type=float, default=0.5, help='the probability for dropout [default: 0.5]')
parser.add_argument('-max-norm', type=float, default=3.0, help='l2 constraint of parameters [default: 3.0]')
parser.add_argument('-embed-path', type=str, default="sgns.weibo.word", help='the path of pretrained embedding')
parser.add_argument('-embed-dim', type=int, default=300, help='number of embedding dimension [default: 300]')
parser.add_argument('-hidden-size', type=int, default=256, help='number of hidden size')
parser.add_argument('-kernel-num', type=int, default=200, help='number of each kind of kernel')
parser.add_argument('-kernel-sizes', type=str, default='2,3,4,5', help='comma-separated kernel size to use for convolution')
parser.add_argument('-static', action='store_true', default=False, help='fix the embedding')
# device
parser.add_argument('-device', type=int, default=-1, help='device to use for iterate data, -1 mean cpu [default: -1]')
parser.add_argument('-no-cuda', action='store_true', default=False, help='disable the gpu')
# option
parser.add_argument('-snapshot', type=str, default=None, help='filename of model snapshot [default: None]')
parser.add_argument('-predict', type=str, default=None, help='predict the sentence given')
parser.add_argument('-test', action='store_true', default=False, help='train or test')
args = parser.parse_args()

batch_size = args.batch_size
max_len = args.max_length
dataset_folder = args.data_path
pretrained_embedding_path = args.embed_path
TEXT, vocab_size, label_size, word_embeddings, train_iter, dev_iter, test_iter, label_field, text_field = load_data.load_dataset(dataset_folder, batch_size, max_len, pretrained_embedding_path)

# learning_rate = 2e-4
# args.lr = learning_rate

# output_size = label_size - 1
# hidden_size = args.hidden_size
embedding_length = args.embed_dim
# load data
print("\nLoading data...")
#text_field = data.Field(lower=True)
##text_field = data.Field(lower=False)
#label_field = data.Field(batch_first=True)
## label_field = data.Field(batch_first=True, sequential=False)
#train_iter, dev_iter = mr(text_field, label_field, device=-1, repeat=False)
## train_iter, dev_iter, test_iter = sst(text_field, label_field, device=-1, repeat=False)


# update args and print
#args.embed_num = len(text_field.vocab)
#args.class_num = len(label_field.vocab) - 1
#args.cuda = (not args.no_cuda) and torch.cuda.is_available(); del args.no_cuda
#args.kernel_sizes = [int(k) for k in args.kernel_sizes.split(',')]
#args.save_dir = os.path.join(args.save_dir, datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
#args.embedding = word_embeddings

args.embed_num = vocab_size
args.class_num = label_size - 1
args.cuda = (not args.no_cuda) and torch.cuda.is_available(); del args.no_cuda
args.kernel_sizes = [int(k) for k in args.kernel_sizes.split(',')]
args.save_dir = os.path.join(args.save_dir, datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
args.embedding = word_embeddings

'''
leaves = {}
label_dict = {}
for line in open(os.path.join(dataset_folder, "train.txt"), 'r', encoding='utf-8').readlines():
    label = line.strip().split("\t")[0]
    leaf = label.strip().split(r"|")[-1]
    leaves[leaf] = 1
    label_dict[label] = 0

params = {}
params['batch_size'] = batch_size
# params['output_size'] = output_size
# params['hidden_size'] = hidden_size
params['vocab_size'] = vocab_size
params['embedding_length'] = embedding_length
params['word_embeddings'] = word_embeddings

model_dir = "model_"+dataset_folder
os.mkdir(model_dir)

with open(model_dir+'/params', 'wb') as f:
        dill.dump(params, f)
with open(model_dir+"/leaves", "wb")as f:
    dill.dump(leaves, f)
with open(model_dir+'/args', 'wb') as f:
    dill.dump(args, f)
with open(model_dir+"/TEXT.Field","wb")as f:
    dill.dump(text_field, f)
with open(model_dir+"/LABEL.Field","wb")as f:
    dill.dump(label_field, f)
with open(model_dir+"/LABEL.Dict","wb")as f:
    dill.dump(label_dict, f)

print("\nParameters:")
for attr, value in sorted(args.__dict__.items()):
    print("\t{}={}".format(attr.upper(), value))
'''

# model
cnn = model.CNN_Text(args)
#cnn = SelfAttention(batch_size, output_size, hidden_size, vocab_size, embedding_length, word_embeddings)
if args.snapshot is not None:
    print('\nLoading model from {}...'.format(args.snapshot))
    cnn.load_state_dict(torch.load(args.snapshot))

if args.cuda:
    cnn = cnn.cuda()

# train or predict
if args.predict is not None:
    label = train.predict(args.predict, cnn, text_field, label_field, args.cuda, leaves)
#    print('\n[Text]  {}\n[Label] {}\n'.format(args.predict, label))
    print(label)
elif args.test:
    try:
        train.eval(test_iter, cnn, label_field, leaves, args) 
    except Exception as e:
        print("\nSorry. The test dataset doesn't  exist.\n")
else:
    print()
    try:
        train.train(train_iter, dev_iter, cnn, label_field, leaves, args)
    except KeyboardInterrupt:
        print('\n' + '-' * 89)
        print('Exiting from training early')
