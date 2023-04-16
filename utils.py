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
      