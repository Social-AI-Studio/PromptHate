import torch
import torch.nn as nn
from transformers import RobertaForMaskedLM

class RobertaPromptModel(nn.Module):
    def __init__(self,label_list):
        super(RobertaPromptModel, self).__init__()
        self.label_word_list=label_list
        self.roberta = RobertaForMaskedLM.from_pretrained('roberta-large')

    def forward(self,tokens,attention_mask,mask_pos,feat=None):
        batch_size = tokens.size(0)
        #the position of word for prediction
        if mask_pos is not None:
            mask_pos = mask_pos.squeeze()
            
        out = self.roberta(tokens, 
                           attention_mask)
        prediction_mask_scores = out.logits[torch.arange(batch_size),
                                          mask_pos]
        
        logits = []
        for label_id in range(len(self.label_word_list)):
            logits.append(prediction_mask_scores[:,
                                                 self.label_word_list[label_id]
                                                ].unsqueeze(-1))
            #print(prediction_mask_scores[:, self.label_word_list[label_id]].shape)
        logits = torch.cat(logits, -1)
        #print(logits.shape)
        return logits
        
    
def build_baseline(opt,label_list):  
    print (label_list)
    return RobertaPromptModel(label_list)

    