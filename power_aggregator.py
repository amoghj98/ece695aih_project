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
    powers = np.zeros(int(len(fileNames) / 2))
    # print(powers.shape)
    cpuOverhead = 0.5
    coolingOverhead = 0.2
    powerKey = 'Total power consumtion:'
    #
    i = 0
    for f in fileNames:
        if f.endswith('.err'):
            with open(os.path.join(filePath, f)) as file:
                lines = reversed(file.readlines())
            for _, line in enumerate(lines, 1):
                if powerKey in line:
                    p = (line.split(':')[3]).split(' ')[1]
                    powers[i] = float(p)
                    i += 1
                    break
    gpuPower = powers.sum()
    totalPower = ((1 + cpuOverhead) * (1 + coolingOverhead)) * gpuPower

    print(f'Approach: {approach}')
    print(f'Total files parsed: {i}')
    print(f'Total GPU Energy Consumption for Approach {approach}: {gpuPower/1e3} kWh')
    print(f'Total Energy Consumption (GPU + CPU + Cooling overhead) for Approach {approach}: {totalPower/1e3} kWh')
    print('')
    return totalPower


if __name__ == "__main__":
    p = 0
    p += powerAggregator('bon_256')
    p += powerAggregator('dvts_256')
    p += powerAggregator('bs_4')
    p += powerAggregator('bs_16')
    p += powerAggregator('bs_64')
    p += powerAggregator('bs_256')
    print(f'Total Consumed Power: {p/1e3} kWh')
