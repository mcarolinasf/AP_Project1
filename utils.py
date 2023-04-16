import matplotlib.pyplot as plt

class PlotRelevantInfo():
  def __init__(self, results):
    self.iterations = 10
    self.acc_and_loss = results[:2]
    self.precision_and_recall = results[2:]
    self.titles = ['Validation Accuracy', 'Validation Loss', 'Precision', 'Recall']

  def plot_results_and_print_means(self, model):
    self.__plot_evolution_of_validation_acc_and_loss()
    self.__plot_avg_precision_and_recall_for_each_class()
    self.__print_avg_validation_accuracy_and_loss()

  def __plot_evolution_of_validation_acc_and_loss(self):
    for result, title in zip(self.acc_and_loss, self.titles[:2]):
      plt.plot(list(zip(*result)))
      plt.xlabel('Epoch')
      plt.ylabel(title)
      plt.title(f'{title} over {self.iterations} iterations')
      plt.show()

  def __plot_avg_precision_and_recall_for_each_class(self):
    for avg, title in zip(self.precision_and_recall, self.titles[2:]):
      avg = np.mean(avg, axis=0)
      plt.bar(np.arange(10), avg)
      plt.xticks(np.arange(10))
      plt.xlabel('Class')
      plt.ylabel(f'Avg {title}')
      plt.title(f'{title} over {self.iterations} iterations')
      plt.show() 
    
  def __print_avg_validation_accuracy_and_loss(self):
    for result, title in zip(self.acc_and_loss, self.titles[:2]):
      lasts = [r[-1] for r in result]
      print(f"Avg {title}: {sum(lasts)/len(lasts)}")

# Plot the training and validation accuracy over time
class PlotData():

    def __init__(self, history):
        self.history = history
        self.__plot_accuracy__()
        self.__plot_loss__()

    def __plot_loss__(self):
       # Plot the training and validation loss over time
        plt.plot(self.history.history['loss'])
        plt.plot(self.history.history['val_loss'])
        plt.title('Model Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['train', 'val'], loc='upper left')
        plt.show()
    
    # Plot the training and validation accuracy over time
    def __plot_accuracy__(self):
        plt.plot(self.history.history['accuracy'])
        plt.plot(self.history.history['val_accuracy'])
        plt.title('Model Accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['train', 'val'], loc='upper left')
        plt.show()


    