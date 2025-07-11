import pickle
import matplotlib.pyplot as plt
from flask import send_file

def get_training_plot():
    with open("vgg_history.pckl", "rb") as f:
        history = pickle.load(f)

    plt.figure(figsize=(10, 4))
    plt.plot(history["accuracy"], label="Accuracy")
    plt.plot(history["val_accuracy"], label="Val Accuracy")
    plt.title("Training Accuracy")
    plt.legend()
    plt.savefig("static/stats.png")
    plt.close()

    return send_file("static/stats.png", mimetype="image/png")
