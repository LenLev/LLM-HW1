import json
import matplotlib.pyplot as plt

with open("output_dir/gpt2-1b-russian/trainer_state.json") as f:
    state = json.load(f)

history = state["log_history"]

train_steps = []
train_loss = []

eval_steps = []
eval_loss = []

for item in history:
    if "loss" in item:
        train_steps.append(item["step"])
        train_loss.append(item["loss"])

    if "eval_loss" in item:
        eval_steps.append(item["step"])
        eval_loss.append(item["eval_loss"])

plt.figure(figsize=(10, 6))

plt.plot(train_steps, train_loss, label="Train loss")
plt.plot(eval_steps, eval_loss, marker="o", label="Eval loss")

plt.xlabel("Step")
plt.ylabel("Loss")
plt.title("Training and Evaluation Loss")
plt.legend()
plt.grid(True)

plt.savefig("output_dir/loss_plot.png", dpi=150)
print("График сохранён: output_dir/loss_plot.png")

