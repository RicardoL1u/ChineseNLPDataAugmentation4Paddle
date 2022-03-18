import unittest
import paddle
import paddle.nn
import numpy as np
import torch
import paddlenlp.transformers 
from transformers import BertTokenizer, BertForMaskedLM

# from transformers import BertForMaskedLM
class TestMask(unittest.TestCase):
    def __init__(self, methodName: str = ...) -> None:
        super().__init__(methodName)
        self.pdbert = paddlenlp.transformers.BertForMaskedLM.from_pretrained('bert-wwm-chinese')
        self.pdtokenizer = paddlenlp.transformers.BertTokenizer.from_pretrained('bert-wwm-chinese')
        self.hftokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
        self.hfbert = BertForMaskedLM.from_pretrained('bert-base-chinese')

    def test_align(self):
        test_str = '巴黎是[MASK]国的首都.'
        pdinput = paddle_input(self.pdtokenizer(test_str))
        hfinput = hugface_input(self.hftokenizer(test_str))
        print(pdinput)
        print(hfinput)
        pdoutput = self.pdbert(**pdinput)
        hfoutput = self.hfbert(**hfinput)
        print(pdoutput)
        print(hfoutput)

        
def paddle_input(tokenizer_output:dict):
    return {k:paddle.to_tensor([v]) for (k, v) in tokenizer_output.items()}
    
def hugface_input(tokenizer_output:dict):
    return {k:torch.tensor([v]) for (k, v) in tokenizer_output.items()}

if __name__ == "__main__":
    unittest.main()