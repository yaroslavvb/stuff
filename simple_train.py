import torch
from torch.utils.data import (DataLoader, SequentialSampler,
                              TensorDataset)

from pytorch_pretrained_bert import OpenAIGPTTokenizer, OpenAIGPTLMHeadModel


def main():
  # 3 examples
  train_dataset = 'small brown fox jumps over the lazy dog\n' \
                  'small brown fox jumps over the lazy dog\n' \
                  'small brown fox jumps over the lazy dog\n'
  tokenizer = OpenAIGPTTokenizer.from_pretrained('openai-gpt',
                                                 special_tokens=[])
  tokenized = [tokenizer.tokenize(t) for t in train_dataset.strip().split('\n')]

  encoded=[tokenizer.convert_tokens_to_ids(t) for t in tokenized]  # 3x8
  dataset = TensorDataset(torch.tensor(encoded))
  sampler = SequentialSampler(dataset)
  dataloader = DataLoader(dataset, sampler=sampler, batch_size=1)
  model = OpenAIGPTLMHeadModel.from_pretrained('openai-gpt')

  optimizer = torch.optim.SGD(model.parameters(), lr = 0.0001, momentum=0.9)

  batch = next(iter(dataloader))
  batch=batch[0]   # dataloader gives [batch] instead of batch...why?
 
  for i in range(20):
    loss = model(input_ids=batch, lm_labels=batch)
    print(loss.detach().numpy())
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

# Should produce this
#  6.134997
#  5.3747735
#  5.164842
#  4.8581843
#  4.346232
#  4.158811
#  3.7503657
#  3.29156
#  2.8858535
#  2.760832
#  2.562772
#  2.0645103
#  1.6837901
#  1.6822727
#  1.5878279
#  1.3873199
#  1.158909
#  0.92595655
#  0.8487712
#  0.82774204


if __name__=='__main__':
  main()
  
