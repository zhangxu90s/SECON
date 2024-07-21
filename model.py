#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 21 09:11:27 2023

@author: zhangxu
"""
import os
import torch.nn as nn
import torch    
import torch.nn.functional as F

class Model(nn.Module):   
    def __init__(self, encoder):
        super(Model, self).__init__()
        self.encoder = encoder

    def forward(self, code_inputs=None, nl_inputs=None): 
                
        if code_inputs is not None:
    
            output1 = self.encoder(code_inputs,attention_mask=code_inputs.ne(1))[0]
            outputs = (output1*code_inputs.ne(1)[:,:,None]).sum(1)/code_inputs.ne(1).sum(-1)[:,None]
            
            return torch.nn.functional.normalize(outputs, p=2, dim=1)

        else:
            output2 = self.encoder(nl_inputs,attention_mask=nl_inputs.ne(1))[0]
            outputs = (output2*nl_inputs.ne(1)[:,:,None]).sum(1)/nl_inputs.ne(1).sum(-1)[:,None]
            return torch.nn.functional.normalize(outputs, p=2, dim=1)

        
class CoModel(nn.Module):   
    def __init__(self, encoder,args):
        super(CoModel, self).__init__()
        self.encoder = encoder
        self.args = args
        '''
        unfreeze_layers = ['layer.10','layer.11']# 'layer.8','layer.9','layer.10','layer.11'
        
        for name ,param in self.encoder.named_parameters():
            param.requires_grad = False
            for ele in unfreeze_layers:
                if ele in name:
                    param.requires_grad = True
                    break      
        '''
        self.poly_m = 8
        self.poly_code_embeddings = nn.Embedding(self.poly_m, 768).to(self.args.device)  
        # https://github.com/facebookresearch/ParlAI/blob/master/parlai/agents/transformer/polyencoder.py#L355
        #torch.nn.init.normal_(self.poly_code_embeddings.weight, 768 ** -0.5)

    def dot_attention(self, q, k, v):
        # q: [bs, poly_m, dim] or [bs, res_cnt, dim]
        # k=v: [bs, length, dim] or [bs, poly_m, dim]
        attn_weights = torch.matmul(q, k.transpose(2, 1)) # [bs, poly_m, length]
        #print("attn_weights",attn_weights.shape)

        attn_weights = F.softmax(attn_weights, -1)
        output = torch.matmul(attn_weights, v) # [bs, poly_m, dim]
        return output
    
    def cross_encoder(self, code_inputs=None, nl_inputs=None):
        # context encoder
        ctx_out = self.encoder(code_inputs,attention_mask=code_inputs.ne(1))[0]  # [bs, length, dim]
        #print("ctx_out",ctx_out.shape)

        poly_code_ids = torch.arange(self.poly_m, dtype=torch.long).to(self.args.device) 
        poly_code_ids = poly_code_ids.unsqueeze(0).expand(ctx_out.shape[0], self.poly_m).to(self.args.device) 

        poly_codes = self.poly_code_embeddings(poly_code_ids).to(self.args.device)  # [bs, poly_m, dim]
        #print("poly_codes",poly_codes.shape)
        embs = self.dot_attention(poly_codes, ctx_out, ctx_out).to(self.args.device)  # [bs, poly_m, dim]

        # response encoder
        cand_emb = self.encoder(nl_inputs,attention_mask=nl_inputs.ne(1))[0][:,0,:] # [bs, dim]
        cand_emb = cand_emb.reshape(cand_emb.shape[0], 1, -1).to(self.args.device)  # [bs, res_cnt, dim]

        ctx_emb = self.dot_attention(cand_emb, embs, embs).to(self.args.device)  # [bs, res_cnt, dim]


        return cand_emb[:,0,:] 
    
    def forward(self, code_inputs=None, nl_inputs=None): 
        v1 = self.cross_encoder(code_inputs=code_inputs, nl_inputs=nl_inputs)
        v2 = self.cross_encoder(code_inputs=nl_inputs, nl_inputs=code_inputs)

        
        return v1, v2
'''              
class CoModel(nn.Module):   
    def __init__(self, encoder,args):
        super(CoModel, self).__init__()
        self.encoder = encoder
        
        unfreeze_layers = ['layer.9','layer.10','layer.11']# 'layer.8','layer.9','layer.10','layer.11'
        
        for name ,param in self.encoder.named_parameters():
            param.requires_grad = False
            for ele in unfreeze_layers:
                if ele in name:
                    param.requires_grad = True
                    break    
        
    def forward(self, code_inputs=None, nl_inputs=None): 
                
    
        output1 = self.encoder(code_inputs,attention_mask=code_inputs.ne(1))[0][:,0,:]
        
        output2 = self.encoder(nl_inputs,attention_mask=nl_inputs.ne(1))[0][:,0,:]
        return output1,output2
'''
    