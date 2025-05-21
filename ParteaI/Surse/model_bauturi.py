import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

df_train = pd.read_csv("../train.csv")
df_test = pd.read_csv("../test.csv")

# separ datele in X si y
X_train = df_train.drop(columns=['bautura_pentru_tine'])
y_train = df_train['bautura_pentru_tine']
X_test = df_test.drop(columns=['bautura_pentru_tine'])
y_test = df_test['bautura_pentru_tine']

# codific variabilele categorice
categoric_columns = X_train.select_dtypes(include=['object']).columns

for col in categoric_columns:
    encoder = LabelEncoder()
    X_train[col] = encoder.fit_transform(X_train[col])
    X_test[col] = encoder.transform(X_test[col])

# antrenez modelul pe setul de antrenare
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# fac predictii pe setul de testare``
predictions = model.predict(X_test)

# metrici de evaluare
print("Acuratete:\n")
print(accuracy_score(y_test, predictions))
print("\n")

print("Raport de clasificare:\n")
print(classification_report(y_test, predictions, zero_division=0))
print("\n")

confusion_mat = confusion_matrix(y_test, predictions)
disp = ConfusionMatrixDisplay(confusion_matrix=confusion_mat,
                            display_labels=model.classes_)
disp.plot(cmap=plt.cm.Blues)
plt.title('Matricea de confuzie')
plt.xticks(rotation=45)
plt.show()
plt.close()