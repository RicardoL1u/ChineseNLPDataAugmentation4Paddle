# ChineseNLPDataAugmentation

Chinese NLP Data Augmentation, BERT Contextual Augmentation, Customized for PaddleNLP

百度飞桨框架下的NLP数据增强 (采用Bert或EDA)



## how this work

### Bert Part

1. Randomly insert several [MASK] tokens or replace some original tokens with [MASK] in the original text

   ```
   before: 时间往往能打败大多数人
   insert: 时间[MASK]往往能打败大多数人
   replace: 时间往往[MASK]打败大多数人
   ```

   > we adopt the jieba to avoid insert [MASK] to one word inside like "时[MASK]间往往能打败大多数人"

2. utilize the ```BertForMaskedLM``` to predict which token the [MASK] should be

3. use the best top k prediction as the result

### EDA Part

TBD

## how to use

1. environment require
   - PaddleNLP
   - PaddlePaddle
   - synonyms // only required in eda part,

2. python augumentor.py --input /path/to/sentences.txt

   > the context in sentences.txt should be like this
   >
   > ```
   > 帮我查一下航班信息
   > 保研没有大多数人想象中的那么难
   > 时间往往能打败大多数人
   > ```
   >
   > one row one sentence

## output

```
input: 帮我查一下航班信息  

output: {'score': [0.15944890677928925, 0.03266862779855728, 0.16812720894813538], 'insert_index': [1, 2, 3], 'token': [6435, 3221, 872], 'token_str': ['请', '是', '你'], 'sequence': '请 是 你 帮 我 查 一 下 航 班 信 息'}

input: 时间往往能打败大多数人 

output: {'score': [0.054044950753450394, 0.925567626953125], 'insert_index': [3, 4], 'token': [1045, 2518], 'token_str': ['光', '往'], 'sequence': '时 光 往 往 能 打 败 大 多 数 人'}
```



