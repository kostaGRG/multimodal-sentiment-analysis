import matplotlib.pyplot as plt
import numpy as np

### In this code script we just collect all the experiment results and plot them on a bar chart ###
languages = ['English', 'Greek', 'German', 'Spanish', 'Russian', 'Hindi']
training_on_english = [73.49, 71.95, 72.2, 70.08, 71.91, 70.27]  
training_on_greek = [71.06, 72.21, 71.12, 71.32, 69.9, 71.17]     
training_on_both = [73.77, 74.72, 70.85, 69.87, 71.63, 71.83]      

bar_width = 0.25  
index = np.arange(len(languages))

# Create bars
plt.bar(index, training_on_english, bar_width, label='Training on English')
plt.bar(index + bar_width, training_on_greek, bar_width, label='Training on Greek')
plt.bar(index + 2 * bar_width, training_on_both, bar_width, label='Training on both')

# Add titles and labels
plt.title('Bar Plot',fontsize=28)
plt.xlabel('Language',fontsize=28)
plt.ylabel('F1 Score',fontsize=28)
plt.xticks(index + bar_width, languages,fontsize=24)  # Positioning the language labels in the center of the group of bars
plt.yticks(fontsize=24)

# Add a legend
plt.legend(fontsize=24)

# Show the plot
plt.show()