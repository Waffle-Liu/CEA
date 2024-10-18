import os 
import random
import torch
import torch.nn as nn
import numpy as np
from copy import deepcopy


def set_seed(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic=True


def setup_optimizer(params=[], lrs=[], weight_decay=0., opt_name='AdamW'):
    opt = getattr(torch.optim, opt_name)
    optimizer = [opt(p, lr=lr, weight_decay=weight_decay) for p, lr in zip(params, lrs)]
    print(f'[INFO]    optimizer: {opt}')

    return optimizer


def copy_model_and_optimizer(model, optimizer):
    """Copy the model and optimizer states for resetting after adaptation."""
    model_state = deepcopy(model.state_dict())
    optimizer_state = [deepcopy(optimizer[i].state_dict()) for i in range(len(optimizer))]

    return model_state, optimizer_state


def load_model_and_optimizer(model, optimizer, model_state, optimizer_state):
    """Restore the model and optimizer states from copies."""
    model.load_state_dict(model_state, strict=True)
    for i in range(len(optimizer)):
        optimizer[i].load_state_dict(optimizer_state[i])
   
    return model, optimizer

    
def configure_model(model):
    model.requires_grad_(False)
    return model


def collect_params(model, train_feature=False, train_all=False, train_LN=True, bias_only=False):
    """ Collect trainable parameters """

    params = []
    names = []
    trainable = ['weight', 'bias']
    if bias_only:
        trainable = ['bias']

    for nm, m in model.named_modules():
        if train_LN: 
            if isinstance(m, nn.LayerNorm) or isinstance(m, nn.GroupNorm):
                for np, p in m.named_parameters():
                    if f"{nm}.{np}" in names:
                            continue
                    if np in trainable:  
                        p.requires_grad = True
                        params.append(p)
                        names.append(f"{nm}.{np}")
                        
        if train_feature:
            if len(str(nm).split('.')) > 1:
                if str(nm).split('.')[1] == 'feature_extractor' or str(nm).split('.')[1] == 'feature_projection':
                    for np, p in m.named_parameters():
                        if f"{nm}.{np}" in names:
                            continue
                        p.requires_grad = True
                        params.append(p)
                        names.append(f"{nm}.{np}")
                        
        if train_all: 
            for np, p in m.named_parameters():
                if f"{nm}.{np}" in names:
                    continue
                if np in trainable:  
                    p.requires_grad = True
                    params.append(p)
                    names.append(f"{nm}.{np}")
            
    return params, names


def decode(outputs, processor, use_lm):
    predicted_ids = torch.argmax(outputs, dim=-1)
    if use_lm:
        beam_width = 5
        alpha = 0.5
        processor.batch_decode(outputs.detach().cpu().numpy(), beam_width=beam_width, alpha=alpha).text
    else:
        transcription = processor.batch_decode(predicted_ids)
    transcription = [s.upper() for s in transcription]
    return transcription


def softmax_entropy(x, dim=2):
    # Entropy of softmax distribution from logits
    return -(x.softmax(dim) * x.log_softmax(dim)).sum(dim)


def mcc_loss(x, reweight=False, dim=2, class_num=32):
    p = x.softmax(dim) # (1, L, D)
    p = p.squeeze(0) # (L, D)
    if reweight: # (1, L, D) * (L, 1) 
        target_entropy_weight = softmax_entropy(x, dim=2).detach().squeeze(0) # instance-wise entropy (1, L, D)
        target_entropy_weight = 1 + torch.exp(-target_entropy_weight) # (1, L)
        target_entropy_weight = x.shape[1] * target_entropy_weight / torch.sum(target_entropy_weight)
        cov_matrix_t = p.mul(target_entropy_weight.view(-1, 1)).transpose(1, 0).mm(p)
    else:    
        cov_matrix_t = p.transpose(1, 0).mm(p) # (D, L) * (L, D) -> (D, D)

    cov_matrix_t = cov_matrix_t / torch.sum(cov_matrix_t, dim=1)
    mcc_loss = (torch.sum(cov_matrix_t) - torch.trace(cov_matrix_t)) / class_num
   
    return mcc_loss


def tc_reg_loss(x, non_blank):
    # temporal coherence reg
    # x (1, C, L)
    x_trans = x.transpose(1,2) # (1, L, C)

    k = 1
    att = torch.matmul(x_trans, x)
    att = torch.softmax(att, dim=-1) # (1, L, L)
    att_x = torch.matmul(att, x_trans) + x_trans # (1, L, C)   
    non_blank = non_blank[:, k:]
    tc_loss = att_x[:,k:][non_blank] - att_x[:, :-k][non_blank]
    tc_loss = torch.norm(tc_loss, p=2, dim=-1).mean(0)

    return tc_loss


def forward_and_adapt(x, model, model_base, optimizer, tc_coef=0.3, em_coef=0.3, step=0, step1=0, temp=1.):
    """Forward and adapt model sequential data."""

    loss = 0
    blank_id = 0

    if step < step1:
        outputs = model(x).logits
        predicted_ids = torch.argmax(outputs, dim=-1)
        non_blank = torch.where(predicted_ids != blank_id, 1, 0).bool() 
        
        if em_coef > 0: 
            e = softmax_entropy(outputs / temp)
            e_non_blank = e[non_blank]
            weight = 1/(1+torch.exp(-e_non_blank))
            e_loss = (weight*e_non_blank).mean()
            loss += e_loss * em_coef
        if 1 - em_coef > 0: 
            c_loss = mcc_loss(outputs / temp, True)
            loss += c_loss * (1 - em_coef)
    
        model.zero_grad()
        loss.backward()
        optimizer[0].step()

    else:
        outputs = model(x).logits
        if 'wav2vec2' in model_base:
            feats = model.wav2vec2.feature_extractor(x)
        elif 'hubert' in model_base:
            feats = model.hubert.feature_extractor(x)
        elif 'wavlm' in model_base:
            feats = model.wavlm.feature_extractor(x)

        predicted_ids = torch.argmax(outputs, dim=-1)
        non_blank = torch.where(predicted_ids != blank_id, 1, 0).bool() 

        if em_coef > 0:     
            e_loss = softmax_entropy(outputs / temp).mean(0).mean()
            loss += e_loss * em_coef
            
        if 1 - em_coef > 0: 
            c_loss = mcc_loss(outputs / temp, True)
            loss += c_loss * (1 - em_coef)
        
        tc_loss = tc_reg_loss(feats, non_blank)
        loss += tc_coef*tc_loss

        model.zero_grad()
        loss.backward()
        optimizer[1].step()

    
    with torch.no_grad():
        outputs = model(x).logits
            
    return outputs


