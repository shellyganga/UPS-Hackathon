import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
df = pd.read_csv("https://raw.githubusercontent.com/shellyganga/UPS-Hackathon-Resources/main/Training%20Set/trainingset_labeled.csv?token=AHT6SSFMJUJ73WF6PSCLIQ3A7B6BG")

sns.countplot(x = 'labels',
              data = df,
              order = df.labels.value_counts().index);
plt.show()