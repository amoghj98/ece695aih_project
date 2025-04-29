import numpy as np
import os


def powerAggregator(approach, filePath='./logs/', cpuOverhead=0.5, coolingOverhead=0.2):
    filePath += approach
    # filePath = "./logs/bon_256"
    try:
        fileNames = os.listdir(filePath)
    except FileNotFoundError:
        print(f'Approach {approach} logs not found. Has execution been completed?')
        return 0
    # print(fileNames)
    # print(len(fileNames))
    latencies = np.zeros(int(len(fileNames) / 2))
    # print(powers.shape)
    timeKey = 'Total runtime:'
    #
    i = 0
    for f in fileNames:
        if f.endswith('.err'):
            with open(os.path.join(filePath, f)) as file:
                lines = reversed(file.readlines())
            for _, line in enumerate(lines, 1):
                if timeKey in line:
                    t = (line.split(':')[3]).split(' ')[1]
                    latencies[i] = float(t)
                    i += 1
                    break
    totalTime = latencies.sum()

    print(f'Approach: {approach}')
    print(f'Total files parsed: {i}')
    print(f'Total runtime latency for {approach}: {totalTime/3600} hr')
    print('')
    return totalTime


if __name__ == "__main__":
    t = 0
    t += powerAggregator('bon_4')
    t += powerAggregator('bon_16')
    t += powerAggregator('bon_64')
    print(f'Total latency: {t/3600} hr')
