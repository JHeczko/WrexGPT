from matplotlib import pyplot as plt
import torch

if __name__ == "__main__":
    history = torch.load("./checkpoints.pt", map_location="gpu" if torch.cuda.is_available() else 'cpu')['history']

    epochs = range(1, len(history['test_loss']) + 1)

    fig, axs = plt.subplots(1, 3, figsize=(18, 5))

    # 1. Wykres Loss
    axs[0].plot(epochs, history['test_loss'], 'b', label='Test Loss')
    axs[0].plot(epochs, history['val_loss'], 'r', label='Val Loss')
    axs[0].set_title('Model Loss')
    axs[0].set_xlabel('Epochs')
    axs[0].set_ylabel('Loss')
    axs[0].legend()
    axs[0].grid(True)

    # 2. Wykres Accuracy
    axs[1].plot(epochs, history['test_acc'], 'b', label='Test Acc')
    axs[1].plot(epochs, history['val_acc'], 'r', label='Val Acc')
    axs[1].set_title('Model Accuracy')
    axs[1].set_xlabel('Epochs')
    axs[1].set_ylabel('Accuracy')
    axs[1].legend()
    axs[1].grid(True)

    # 3. Wykres Perplexity
    axs[2].plot(epochs, history['test_ppl'], 'b', label='Test PPL')
    axs[2].plot(epochs, history['val_ppl'], 'r', label='Val PPL')
    axs[2].set_title('Model Perplexity (PPL)')
    axs[2].set_xlabel('Epochs')
    axs[2].set_ylabel('PPL')
    axs[2].legend()
    axs[2].grid(True)

    plt.tight_layout()
    plt.savefig("./history_plots.png")
    plt.show()