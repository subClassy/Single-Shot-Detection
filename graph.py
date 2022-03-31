import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    training_losses = []
    val_losses = []
    with open('log.txt', encoding="utf8") as file:
        while line := file.readline():
            if "loss:" in line.split():
                line_info = line.split()
                if "train" in line_info:
                    training_losses.append(float(line_info[-1]))
                elif "validation" in line_info:
                    val_losses.append(float(line_info[-1]))
    
    
    plt.plot(range(len(training_losses)), training_losses)
    plt.title('Trend for training loss across epochs')
    plt.xlabel('epochs')
    plt.ylabel('Training Loss')
    plt.show()

    plt.plot(range(len(val_losses)), val_losses)
    plt.title('Trend for validation loss across epochs')
    plt.xlabel('epochs')
    plt.ylabel('Validation Loss')
    plt.show()
