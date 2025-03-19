import re

file_path = '/path/to/tensorboard/logfile'

patterns = {
    "Car AP_R40@0.70, 0.70, 0.70": re.compile(
        r"Car AP_R40@0\.70, 0\.70, 0\.70:[\s\S]*?3d\s+AP:\s*([\d\.]+),\s*([\d\.]+),\s*([\d\.]+)",
        re.DOTALL
    ),
    "Pedestrian AP_R40@0.50, 0.50, 0.50": re.compile(
        r"Pedestrian AP_R40@0\.50, 0\.50, 0\.50:[\s\S]*?3d\s+AP:\s*([\d\.]+),\s*([\d\.]+),\s*([\d\.]+)",
        re.DOTALL
    ),
    "Cyclist AP_R40@0.50, 0.50, 0.50": re.compile(
        r"Cyclist AP_R40@0\.50, 0\.50, 0\.50:[\s\S]*?3d\s+AP:\s*([\d\.]+),\s*([\d\.]+),\s*([\d\.]+)",
        re.DOTALL
    )
}

# the query range of saved epoch results 
epoch_range = range(31, 61)

default_weights = {
    "Car AP_R40@0.70, 0.70, 0.70": 1 / 3,
    "Pedestrian AP_R40@0.50, 0.50, 0.50": 1 / 3,
    "Cyclist AP_R40@0.50, 0.50, 0.50": 1 / 3
}

epoch_metrics = {epoch: {metric: None for metric in patterns.keys()} for epoch in epoch_range}

with open(file_path, 'r') as file:
    content = file.read()

epoch_blocks = re.split(r"[*]{15} EPOCH (\d+) EVALUATION [*]{15}", content)


for i in range(1, len(epoch_blocks), 2):
    epoch = int(epoch_blocks[i])  
    if epoch not in epoch_range:
        continue  
    block = epoch_blocks[i + 1]  
    for metric, pattern in patterns.items():
        match = pattern.search(block)
        if match:
            values = [float(v) for v in match.groups()] 
            average = sum(values) / len(values)  
            epoch_metrics[epoch][metric] = average

epoch_scores = {}
for epoch, metrics in epoch_metrics.items():
    available_metrics = {k: v for k, v in metrics.items() if v is not None}  
    if available_metrics:  
        total_weight = sum(default_weights[k] for k in available_metrics.keys())  
        normalized_weights = {k: default_weights[k] / total_weight for k in available_metrics.keys()}  
        score = sum(available_metrics[metric] * normalized_weights[metric] for metric in available_metrics.keys())
        epoch_scores[epoch] = score


best_epoch, best_score = max(epoch_scores.items(), key=lambda x: x[1])
worst_epoch, worst_score = min(epoch_scores.items(), key=lambda x: x[1])


print("Best Epoch Result:")
for epoch, score in sorted(epoch_scores.items(), key=lambda x: x[1], reverse=True):
    print(f"Epoch {epoch}: Score = {score:.2f}")

print(f"\nBest Epoch: {best_epoch}, Score = {best_score:.2f}")
print(f"Worst Epoch: {worst_epoch}, Score = {worst_score:.2f}")


