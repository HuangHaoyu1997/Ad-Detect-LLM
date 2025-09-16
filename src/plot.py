import json
import matplotlib.pyplot as plt
import numpy as np

with open('output/v1-20250915-102750/logging.jsonl', 'r') as f:
    data = [json.loads(line) for line in f]

losses, eval_losses, epochs, eval_epochs = [], [], [], []
for item in data:
    if 'loss' in item:
        losses.append(item['loss'])
        epochs.append(item['epoch'])
    if 'eval_loss' in item:
        eval_losses.append(item['eval_loss'])
        eval_epochs.append(item['epoch'])


plt.plot(epochs, losses, label='train')
plt.plot(eval_epochs, eval_losses, label='eval')
plt.grid()
plt.xlabel('Epochs', fontsize=14)
plt.ylabel('Loss', fontsize=14)
plt.legend(fontsize=14)
plt.savefig('./assets/v1-20250915-102750-loss.png')
plt.show()