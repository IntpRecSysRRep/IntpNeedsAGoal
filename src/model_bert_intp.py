import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_pretrained_bert import BertForSequenceClassification


class BertClassifier(nn.Module):
    def __init__(self, pre_model, DEVICE):
        super(BertClassifier, self).__init__()
        self.pre_model = pre_model
        self.DEVICE = DEVICE

    def forward(self, x, segments_ids, input_masks):
        '''
            x: Tensor(batch_size, text_length, dim)
        '''
        logit = self.pre_model(inputs_embeds=x,
                               token_type_ids=segments_ids,
                               attention_mask=input_masks)
        return logit


if __name__ == "__main__":
    pre_model = BertForSequenceClassification.from_pretrained('bert-base-uncased',
                                                              cache_dir='../result/', num_labels=2)
    path_save = "../result/SST2/"
    WEIGHTS_NAME = "bert.bin"
    filename = os.path.join(path_save, WEIGHTS_NAME)
    pre_model.load_state_dict(torch.load(filename))
    model = BertClassifier(pre_model, torch.device('cpu'))
