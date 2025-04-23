import matplotlib.pyplot as plt
import sys

# Example data: replace these with your actual arrays
accs = {'llama3-1-8bi':{'math':0.229, 'gpqa':0.259},
        'llama3-1-70bi':{'math':0.367, 'gpqa':0.442},
        'gpt4':{'math':0.4, 'gpqa':0.424},
        'gpt4o':{'math':0.533, 'gpqa':0.492},
        'claude3-5':{'math':0.569, 'gpqa':0.553},
        'deepseek-v3':{'math':0.649, 'gpqa':0.565},
        'gemini1-5':{'math':0.704, 'gpqa':0.572},
        'deepseek-r1':{'math':0.931, 'gpqa':0.717},
        }


generations_bon = [1, 2, 4, 8, 16, 32, 64, 128]
accuracy_bon = [28.8, 35.6, 40.4, 42.2, 45.4, 45.8, 47.2, 48.2, 49.6]

# Create the plot
fig, ax = plt.subplots(figsize=(8, 5))

math_accs = [100*v['math'] for v in accs.values()]
gpqa_accs = [100*v['gpqa'] for v in accs.values()]
ax.plot(accs.keys(), math_accs, label='MATH-5')
ax.plot(accs.keys(), gpqa_accs, label='GPQA-Diamond')
plt.xticks(rotation=90)
ax.set_xlabel('Model')
ax.set_ylabel('Accuracy (%)')
ax.set_title('Accuracy Trend for SoTA LLM models on MATH-5 and GPQA-Diamond Benchmarks')
ax.legend()

# ax.plot(generations_bon, accuracy_bon, marker='o', linestyle='-')
# ax.set_xscale('log', base=2)          # set x-axis to log scale (base 2)
# ax.set_xlabel('Number of Generations per problem')
# ax.set_ylabel('MATH-500 Accuracy (%)')
# # ax.set_title('Accuracy vs. Generations')
# ax.grid(True, which='both', ls='--', lw=0.5)

# Optionally: tighten layout and show
fig.tight_layout()
plt.savefig('plt.png')
# plt.show()