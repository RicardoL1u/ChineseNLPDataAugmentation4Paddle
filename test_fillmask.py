import unittest
import paddle
import paddle.nn
import numpy as np
import torch
from paddlenlp.transformers import BertModel, BertTokenizer,BertForMaskedLM
# from transformers import BertForMaskedLM
class TestMask(unittest.TestCase):
    def __init__(self, methodName: str = ...) -> None:
        super().__init__(methodName)
        self.model = BertForMaskedLM.from_pretrained('bert-base-chinese')
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
    def test_fill(self):
        test_str = '在这篇文章中，我会努力去概述[MASK]当前用于文本数据增强的方法.'
        model_inputs = self.tokenizer(test_str)
        input_ids = model_inputs["input_ids"]
        # print(type(model_inputs),model_inputs)
        model_outputs = self.model(**{k:paddle.to_tensor([v]) for (k,v) in model_inputs.items()})
        print("\n========================This Main======================================\n")
        top_k = 5
        outputs = model_outputs
        masked_list = np.squeeze(self.tokenizer.mask_token_id == np.array(input_ids))
        masked_index = paddle.nonzero(paddle.to_tensor(masked_list,dtype='int32'), as_tuple=False).squeeze(axis=-1).tolist()
        # Fill mask pipeline supports only one ${mask_token} per sample
        outputs = outputs.numpy()
        logits = paddle.to_tensor(outputs[0, masked_index, :])
        probs = paddle.nn.functional.softmax(logits,axis=-1)
        values, predictions = probs.topk(top_k)

        result = []
        for i, (_values, _predictions) in enumerate(zip(values.tolist(), predictions.tolist())):
            row = []
            for v, p in zip(_values, _predictions):
                # Copy is important since we're going to modify this array in place
                tokens = np.array(input_ids).copy()
                tokens[masked_index[i]] = p
                # Filter padding out:
                tokens = tokens[np.where(tokens != self.tokenizer.pad_token_id)]
                # Originally we skip special tokens to give readable output.
                # For multi masks though, the other [MASK] would be removed otherwise
                # making the output look odd, so we add them back
                sequence = self.tokenizer.convert_tokens_to_string(self.tokenizer.convert_ids_to_tokens(tokens,skip_special_tokens=(values.shape[0] == 1)))
                proposition = {"score": v, "token": p, "token_str": self.tokenizer.convert_ids_to_tokens(p), "sequence": sequence}
                row.append(proposition)
            result.append(row)

        print(result)

if __name__ == "__main__":
    unittest.main()