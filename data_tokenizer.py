def tokenize(data, tokenizer, dataset_name, ratio=1):

    if dataset_name == 'snli':
        premise = data['premise'][:int(len(data) * ratio)]
        hypothesis = data['hypothesis'][:int(len(data) * ratio)]
        label = data['label'][:int(len(data) * ratio)]

        return {
                'tokenized_input': tokenizer(premise, hypothesis, max_length=512, return_attention_mask=True,
                                             truncation=True, padding='max_length'),
                'tokenized_target': label
                }
    else:
        raise ValueError('No such dataset: {}'.format(dataset_name))