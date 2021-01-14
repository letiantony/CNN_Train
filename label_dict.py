import dill
import codecs
import sys

label_dict = dict()
for line in codecs.open(sys.argv[1], "r", "utf8"):
    label = line.strip()
    label_dict[label] = 0 
with open("model_anquanbu/LABEL.Dict","wb")as f:
    dill.dump(label_dict, f)


