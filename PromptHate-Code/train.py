import os
import time 
import torch
import torch.nn as nn
import utils
import torch.nn.functional as F
import config
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
from transformers import get_linear_schedule_with_warmup,AdamW
from dataset import Multimodal_Data

def bce_for_loss(logits,labels):
    loss=nn.functional.binary_cross_entropy_with_logits(logits, labels)
    loss*=labels.size(1)
    return loss

def compute_auc_score(logits,label):
    bz=logits.shape[0]
    logits=logits.cpu().numpy()
    label=label.cpu().numpy()
    auc=roc_auc_score(label,logits,average='weighted')*bz
    return auc

def compute_score(logits,labels):
    #print (logits,logits.shape)
    logits=torch.max(logits,1)[1]
    #print (logits)
    one_hot=torch.zeros(*labels.size()).cuda()
    one_hot.scatter_(1,logits.view(-1,1),1)
    score=one_hot * labels
    return score.sum().float()

def compute_scaler_score(logits,labels):
    #print (logits,logits.shape)
    logits=torch.max(logits,1)[1]
    labels=labels.squeeze(-1)
    score=(logits==labels).int()
    #print (score.sum(),labels,logits)
    return score.sum().float()


def log_hyperpara(logger,opt):
    dic = vars(opt)
    for k,v in dic.items():
        logger.write(k + ' : ' + str(v))
        
def train_for_epoch(opt,model,train_loader,test_loader):
    #initialization of saving path
    if opt.SAVE:
        model_path=os.path.join('../models',
                          '_'.join([opt.MODEL,opt.DATASET]))
        if os.path.exists(model_path)==False:
            os.mkdir(model_path)
    #multi-qeury configuration
    if opt.MULTI_QUERY:
        from transformers import RobertaTokenizer
        tokenizer = RobertaTokenizer.from_pretrained('roberta-large')
    #initialization of logger
    log_path=os.path.join(opt.DATASET)
    if os.path.exists(log_path)==False:
        os.mkdir(log_path)
    logger=utils.Logger(os.path.join(log_path,str(opt.SAVE_NUM)+'.txt'))  
    log_hyperpara(logger,opt)
    logger.write('Length of training set: %d, length of testing set: %d' %
                 (len(train_loader.dataset),len(test_loader.dataset)))
    
    if opt.MODEL=='pbm':
        #initialization of optimizer
        params = {}
        for n, p in model.named_parameters():
            if opt.FIX_LAYERS > 0:
                if 'encoder.layer' in n:
                    try:
                        layer_num = int(n[n.find('encoder.layer') + 14:].split('.')[0])
                    except:
                        print(n)
                        raise Exception("")
                    if layer_num >= opt.FIX_LAYERS:
                        print('yes', n)
                        params[n] = p
                    else:
                        print('no ', n)
                elif 'embeddings' in n:
                    print('no ', n)
                else:
                    print('yes', n)
                    params[n] = p
            else:
                params[n] = p
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in params.items() if not any(nd in n for nd in no_decay)],
                "weight_decay": opt.WEIGHT_DECAY,
            },
            {
                "params": [p for n, p in params.items() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
    
        optim = AdamW(
            optimizer_grouped_parameters,
            lr=opt.LR_RATE,
            eps=opt.EPS,
        )
    else:
        params=model.parameters()
        optim=AdamW(params,
                    lr=1e-5,
                    eps=1e-8
                   )
    num_training_steps=len(train_loader) * opt.EPOCHS
    scheduler=\
    get_linear_schedule_with_warmup(optim,
                                    num_warmup_steps=0,
                                    num_training_steps=num_training_steps
                                   )
    loss_fn=torch.nn.BCELoss()
    loss_fct = nn.KLDivLoss(log_target=True)
    #strat training
    record_auc=[]
    record_acc=[]
    for epoch in range(opt.EPOCHS):
        model.train(True)
        total_loss=0.0
        scores=0.0
        for i,batch in enumerate(train_loader):
            #break
            cap=batch['cap_tokens'].long().cuda()
            label=batch['label'].float().cuda().view(-1,1)
            mask=batch['mask'].cuda()
            target=batch['target'].cuda()
            feat=None
            if opt.MODEL=='pbm':
                mask_pos=batch['mask_pos'].cuda()
                logits=model(cap,mask,mask_pos,feat)
                if opt.FINE_GRIND:
                    attack=batch['attack'].cuda()#B,6
                    #print (logits)
                    #print (attack)
                    #logits=logits*attack
                    logits[:,1]=torch.sum(logits[:,1:],dim=1)
                    logits=logits[:,:2]
                #print (logits.shape,attack.shape)
            elif opt.MODEL=='roberta':
                if opt.UNIMODAL==False:
                    feat=batch['feat'].cuda()
                logits=model(cap,mask,feat)
            loss=bce_for_loss(logits,target)
            #print (logits,target)
            batch_score=compute_score(logits,target)
            scores+=batch_score
            
            print ('Epoch:',epoch,'Iteration:', i, loss.item(),batch_score)
            loss.backward()
            optim.step()
            scheduler.step()
            optim.zero_grad()
            
            total_loss+=loss
            
        
        model.train(False)
        scores/=len(train_loader.dataset)
        if opt.MULTI_QUERY==False:
            eval_acc,eval_auc=eval_model(opt,model,test_loader)
        else:
            eval_acc,eval_auc=eval_multi_model(opt,model,tokenizer)
        record_auc.append(eval_auc)
        record_acc.append(eval_acc)
        logger.write('Epoch %d' %(epoch))
        logger.write('\ttrain_loss: %.2f, accuracy: %.2f' % (total_loss, 
                                                             scores*100.0))
        logger.write('\tevaluation auc: %.2f, accuracy: %.2f' % (eval_auc, 
                                                                 eval_acc))
    max_idx=sorted(range(len(record_auc)),
                   key=lambda k: record_auc[k]+record_acc[k],
                   reverse=True)[0]
    logger.write('Maximum epoch: %d' %(max_idx))
    logger.write('\tevaluation auc: %.2f, accuracy: %.2f' % (record_auc[max_idx], 
                                                             record_acc[max_idx]))
        
def eval_model(opt,model,test_loader):
    scores=0.0
    auc=0.0
    len_data=len(test_loader.dataset)
    print ('Length of test set:',len_data)
    total_logits=[]
    total_labels=[]
    for i,batch in enumerate(test_loader):
        with torch.no_grad():
            cap=batch['cap_tokens'].long().cuda()
            label=batch['label'].float().cuda().view(-1,1)
            mask=batch['mask'].cuda()
            target=batch['target'].cuda()
            feat=None
            if opt.MODEL=='pbm':
                mask_pos=batch['mask_pos'].cuda()
                logits=model(cap,mask,mask_pos,feat)
                if opt.FINE_GRIND:
                    #attack=batch['attack'].cuda()#B,6
                    #logits=logits*attack
                    logits[:,1]=torch.sum(logits[:,1:],dim=1)
                    logits=logits[:,:2]
                
            elif opt.MODEL=='roberta':
                if opt.UNIMODAL==False:
                    feat=batch['feat'].cuda()
                logits=model(cap,mask,feat)
            
            batch_score=compute_score(logits,target)
            scores+=batch_score
            norm_logits=F.softmax(logits,dim=-1)[:,1].unsqueeze(-1)
            
            total_logits.append(norm_logits)
            total_labels.append(label)
    total_logits=torch.cat(total_logits,dim=0)
    total_labels=torch.cat(total_labels,dim=0)
    print (total_logits.shape,total_labels.shape)
    auc=compute_auc_score(total_logits,total_labels)
    #print (auc)
    return scores*100.0/len_data,auc*100.0/len_data

def eval_multi_model(opt,model, tokenizer):
    num_queries=opt.NUM_QUERIES
    labels_record={}
    logits_record={}
    prob_record={}
    for k in range(num_queries):
        test_set=Multimodal_Data(opt,tokenizer,opt.DATASET,'test')
        test_loader=DataLoader(test_set,
                               opt.BATCH_SIZE,
                               shuffle=False,
                               num_workers=1)
        len_data=len(test_loader.dataset)
        print ('Length of test set:',len_data,'Query:',k)
        for i,batch in enumerate(test_loader):
            with torch.no_grad():
                cap=batch['cap_tokens'].long().cuda()
                label=batch['label'].float().cuda().view(-1,1)
                mask=batch['mask'].cuda()
                mask_pos=batch['mask_pos'].cuda()
                logits=model(cap,mask,mask_pos)
                if opt.FINE_GRIND:
                    #attack=batch['attack'].cuda()#B,6
                    #logits=logits*attack
                    logits[:,1]=torch.sum(logits[:,1:],dim=1)
                    logits=logits[:,:2]
                target=batch['target'].cuda()
                img=batch['img']
                norm_prob=F.softmax(logits,dim=-1)
                norm_logits=norm_prob[:,1].unsqueeze(-1)
                
                bz=cap.shape[0]
                for j in range(bz):
                    cur_img=img[j]
                    cur_logits=norm_logits[j:j+1]
                    #should normalize to the same scale
                    cur_prob=norm_prob[j:j+1]
                    if k==0:
                        cur_label=label[j:j+1]
                        labels_record[cur_img]=cur_label
                        logits_record[cur_img]=cur_logits
                        prob_record[cur_img]=cur_prob
                    else:
                        logits_record[cur_img]+=cur_logits
                        prob_record[cur_img]+=cur_prob
    labels=[] 
    logits=[]
    probs=[]
    for name in labels_record.keys():
        labels.append(labels_record[name])
        logits.append(logits_record[name]/num_queries)
        probs.append(prob_record[name]/num_queries)
            
    logits=torch.cat(logits,dim=0)
    labels=torch.cat(labels,dim=0)
    probs=torch.cat(probs,dim=0)
    
    scores=compute_scaler_score(probs,labels)
    auc=compute_auc_score(logits,labels)
    #print (auc)
    return scores*100.0/len_data,auc*100.0/len_data