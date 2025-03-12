import seaborn as sns
import matplotlib.pyplot as plt

# Load a sample dataset
tips = sns.load_dataset("tips")

# Create a basic histogram
sns.histplot(data=tips, x="total_bill")

# Show the plot
plt.show()

# create a histogram with a kde.
sns.histplot(data=tips, x="total_bill", kde=True)
plt.show()

#create a histogram with hue.
sns.histplot(data=tips, x="total_bill", hue="sex")
plt.show()