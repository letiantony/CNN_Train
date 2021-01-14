import os
import sys
import torch
import torch.autograd as autograd
import torch.nn.functional as F


def train(train_iter, dev_iter, model, label_field, leaves, args):
    if args.cuda:
        model.cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    steps = 0
    best_acc = 0
    last_step = 0
    multi_label_loss = torch.nn.MultiLabelSoftMarginLoss()
    #multi_label_loss = torch.nn.CrossEntropyLoss()
    for epoch in range(1, args.epochs+1):
        for batch in train_iter:
            model.train()
            feature, target = batch.text, batch.label
            feature.data.t_()
            target.data.sub_(1)  # batch first, index align
            if args.cuda:
                feature, target = feature.cuda(), target.cuda()

            optimizer.zero_grad()
            logit = model(feature)

            #print('logit vector', logit.size())
            #print('target vector', target.size())
            # loss = F.cross_entropy(logit, target)
            if args.cuda:
                target_onehot = torch.zeros(target.shape[0], args.class_num).cuda().scatter_(1, target, 1)
            else:
                target_onehot = torch.zeros(target.shape[0], args.class_num).scatter_(1, target, 1)
            target_onehot[:, 0] = 0
            loss = multi_label_loss(logit, target_onehot)
            loss.backward()
            optimizer.step()

            steps += 1
            if steps % args.log_interval == 0:
                _, sorted_idx = torch.sort(logit, 1, descending=True)
                pred = torch.where(target > 0, sorted_idx[:, :target.shape[1]], target)
                if args.cuda:
                    pred_onehot = torch.zeros(target.shape[0], args.class_num).cuda().fill_(-1).scatter_(1, pred, 1)
                else:
                    pred_onehot = torch.zeros(target.shape[0], args.class_num).fill_(-1).scatter_(1, pred, 1)
                pred_onehot[:, 0] = -1
                tag_num = torch.sum(target_onehot, 1)
                corrects = (torch.sum((pred_onehot == target_onehot), 1).float() / tag_num).sum()
                # corrects = (torch.max(logit, 1)[1].view(target.size()).data == target.data).sum()
                accuracy = 100.0 * corrects/batch.batch_size
                sys.stdout.write(
                    '\rBatch[{}] - loss: {:.6f}  acc: {:.4f}%({}/{})'.format(steps, 
                                                                             loss.item(), 
                                                                             accuracy,
                                                                             corrects,
                                                                             batch.batch_size))
                sys.stdout.flush()
            if steps % args.test_interval == 0:
                dev_acc = eval(dev_iter, model, label_field, leaves, args)
                if dev_acc > best_acc:
                    best_acc = dev_acc
                    last_step = steps
                    if args.save_best:
                        save(model, args.save_dir, 'best', steps)
                        bestFolder = ','.join([args.save_dir, 'best', str(steps)])
                else:
                    if steps - last_step >= args.early_stop:
                        print('early stop by {} steps.'.format(args.early_stop))
                        print('best test acc {}, save model {}'.format(best_acc, bestFolder)) 
                        return 
            elif steps % args.save_interval == 0:
                save(model, args.save_dir, 'snapshot', steps)
    print('best test acc {}, save model {}'.format(best_acc, bestFolder))

def get_type(sorted_idx, label_field, leaves):
    leaf = None
    for i in sorted_idx:
        if label_field.vocab.itos[i+1] in leaves:
            leaf = label_field.vocab.itos[i+1]
            return leaf
    return leaf

def eval(data_iter, model, label_field, leaves, args):
    model.eval()
    #size = len(data_iter.dataset)
    size = 0
    corrects, avg_loss = 0, 0
    multi_label_loss = torch.nn.MultiLabelSoftMarginLoss()
    
    for batch in data_iter:
        feature, target = batch.text, batch.label
        feature.data.t_(), target.data.sub_(1)  # batch first, index align
        if args.cuda:
            feature, target = feature.cuda(), target.cuda()

        logit = model(feature)
        if args.cuda:
            target_onehot = torch.zeros(target.shape[0], args.class_num).cuda().scatter_(1, target, 1)
        else:
            target_onehot = torch.zeros(target.shape[0], args.class_num).scatter_(1, target, 1)
        target_onehot[:, 0] = 0
        if args.cuda:
            target_onehot = target_onehot.cuda()

        loss = multi_label_loss(logit, target_onehot)

        avg_loss += loss.item()

        sorted_score, sorted_idx = torch.sort(logit, 1, descending=True)

        normalize_op = torch.nn.Softmax(dim=1)
        prob = normalize_op(sorted_score)

        for text_id in range(len(target)):
            #for tag_pos in range(10):
            #    print(label_field.vocab.itos[sorted_idx[text_id][tag_pos] + 1], prob[text_id][tag_pos])
            #print(target[text_id], sorted_idx[text_id])
            gold_leaf = get_type(target[text_id], label_field, leaves)
            if gold_leaf is None:
                continue
            pred_leaf = get_type(sorted_idx[text_id], label_field, leaves)
            #print(gold_leaf, pred_leaf)
            if gold_leaf == pred_leaf:
                corrects += 1
            size += 1
        #topk = 5
        ##pred = torch.where(target > 0, sorted_idx[:, :target.shape[1]], target)
        #pred = sorted_idx[:, :topk]
        #pred_onehot = torch.zeros(target.shape[0], args.class_num).cuda().fill_(-1).scatter_(1, pred, 1)
        #pred_onehot[:, 0] = -1
        ##tag_num = torch.sum(target_onehot, 1)
        #tag_num = topk
        #corrects += (torch.sum((pred_onehot == target_onehot), 1).float() / tag_num).sum()

        ## corrects += (torch.max(logit, 1)
        ##              [1].view(target.size()).data == target.data).sum()

    avg_loss /= size
    accuracy = 100.0 * corrects/size
    print('\nEvaluation - loss: {:.6f}  acc: {:.4f}%({}/{}) \n'.format(avg_loss, 
                                                                       accuracy, 
                                                                       corrects, 
                                                                       size))
    return accuracy


def predict(text, model, text_field, label_field, cuda_flag, leaves):
    assert isinstance(text, str)
    model.eval()
    # text = text_field.tokenize(text)
    #text = text_field.preprocess(text)
    text = [[text_field.vocab.stoi[x] for x in text]]
    x = text_field.tensor_type(text)
    x = autograd.Variable(x, volatile=True)
    if cuda_flag:
        x = x.cuda()
    print(x)
    output = model(x, batch_size=len(text))
    _, sorted_idx = torch.sort(output, 1, descending=True)
    pred_leaf = get_type(sorted_idx[0], label_field, leaves)
    #return label_feild.vocab.itos[predicted.data[0][0]+1]
    #return pred_leaf
    return [label_field.vocab.itos[label_idx+1] for label_idx in sorted_idx[0][:10]]


def save(model, save_dir, save_prefix, steps):
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    save_prefix = os.path.join(save_dir, save_prefix)
    save_path = '{}_steps_{}.pt'.format(save_prefix, steps)
    torch.save(model.state_dict(), save_path)
