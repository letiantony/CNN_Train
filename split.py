import codecs
import random
import sys
import os

input_path = sys.argv[1]
data_dir = sys.argv[2]
os.mkdir(data_dir)

train_writer = codecs.open(os.path.join(data_dir,"train.txt"),"w","utf8")
dev_writer = codecs.open(os.path.join(data_dir,"dev.txt"),"w","utf8")
test_writer = codecs.open(os.path.join(data_dir,"test.txt"),"w","utf8")

label_dict = dict()
case_num = 0

for line in codecs.open(input_path, "r", "utf8"):
    linelist = line.strip().split("\t")
    if len(linelist) != 2:
        continue
    text = linelist[0]
    label = linelist[1]
    if label not in label_dict:
        label_dict[label] = list()
    label_dict[label].append(text)
    case_num += 1

label_num = len(label_dict)
avg_num = case_num//label_num
upper_bound = (case_num//label_num) * 5
bottom_bound = (case_num//label_num)//2 

def balance(temp_list):
    raw_list = temp_list
    if len(temp_list) > upper_bound:
        random.shuffle(temp_list)
        return temp_list[:upper_bound]
    if len(temp_list) < bottom_bound:
        while len(temp_list) < bottom_bound:
            temp_list.appen(random.choice(raw_list))
        return temp_list

train_list = list()
dev_list = list()
test_list = list()

for label in label_dict:
    temp_list = label_dict.get(label)
    list_len = len(temp_list)
    if list_len< 10:
        continue
    temp_list = balance(temp_list)
    dev_num = int(list_len*0.8)
    test_num = int(list_len*0.9)
    random.shuffle(temp_list)
    for i in range(0,dev_num):
        train_list.append(label+"\t"+temp_list[i])
    for i in range(dev_num,test_num):
        dev_list.append(label+"\t"+temp_list[i])
    for i in range(test_num,list_len):
        test_list.append(label+"\t"+temp_list[i])

random.shuffle(train_list)
random.shuffle(dev_list)
random.shuffle(test_list)
for x in train_list:
    train_writer(x+"\n")
for x in dev_list:
    dev_writer(x+"\n")
for x in test_list:
    test_writer(x+"\n")
train_writer.close()
dev_writer.close()
test_writer.close()
