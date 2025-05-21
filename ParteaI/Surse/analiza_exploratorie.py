import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style = "whitegrid")

df_train = pd.read_csv("../train.csv")
df_test = pd.read_csv("../test.csv")

# dimensiunile seturilor de date
print("train:", df_train.shape)
print("test:", df_test.shape)
print("\n\n")

# arat ca nu avem valori lipsa
print("Valori lipsa in train:\n", df_train.isnull().sum())
print("\n\n")
print("Valori lipsa in test:\n", df_test.isnull().sum())
print("\n\n")

# statistici descriptive
print("Descriere numerica train:\n", df_train.describe())
print("\n")

# selectez doar coloanele de tip obiect si vedem cate valori unice are fiecare
print("Numar de elemente pe categorie:\n", 
        df_train.select_dtypes(include = ['object']).nunique())
print("\n")

print("Descriere valori categorice train:\n")
columns = df_train.select_dtypes(include = ['object']).columns
for col in columns:
    print(f"\nCategoria: {col}")
    print("\nNumar aparitii elemente:")
    print(df_train[col].value_counts())
print("\n")

print("Descriere numerica test:\n", df_test.describe())
print("\n")

# selectez doar coloanele de tip obiect si vedem cate valori unice are fiecare
print("Numar de elemente pe categorie:\n",
        df_test.select_dtypes(include = ['object']).nunique())
print("\n")

print("Descriere valori categorice test:\n")
columns = df_test.select_dtypes(include = ['object']).columns
for col in columns:
    print(f"\nCategoria: {col}")
    print("\nNumar aparitii elemente:")
    print(df_test[col].value_counts())
print("\n")

# distributii variabile
# histograme pentru valori numerice train
numeric_columns = ['nr_ore_somn', 'fericire', 'stres', 'temperatura']
for col in numeric_columns:
    plt.figure()
    sns.histplot(df_train[col], kde = True)
    plt.title(f"Distributia {col}")
    plt.show()
    plt.close()

# countplot pentru variabile categorice train
categorical_columns = ['alcool', 'cat_de_acru', 'cantitate_zahar',
                     'vreme', 'energie', 'multa_treaba', 'bautura_pentru_tine']
for col in categorical_columns:
    plt.figure()
    sns.countplot(x = col, data = df_train)
    plt.title(f"Distributia {col}")
    plt.xticks(rotation = 45)
    plt.show()
    plt.close()

# histograme pentru valori numerice test
numeric_columns = ['nr_ore_somn', 'fericire', 'stres', 'temperatura']
for col in numeric_columns:
    plt.figure()
    sns.histplot(df_test[col], kde = True)
    plt.title(f"Distributia {col}")
    plt.show()
    plt.close()

# countplot pentru variabile categorice test
categorical_columns = ['alcool', 'cat_de_acru', 'cantitate_zahar',
                    'vreme', 'energie', 'multa_treaba', 'bautura_pentru_tine']
for col in categorical_columns:
    plt.figure()
    sns.countplot(x = col, data = df_test)
    plt.title(f"Distributia {col}")
    plt.xticks(rotation = 45)
    plt.show()
    plt.close()

# detectarea outlierilor train
for col in numeric_columns:
    plt.figure()
    sns.boxplot(x = df_train[col])
    plt.title(f"Boxplot {col}")
    plt.show()
    plt.close()

# detectarea outlierilor test
for col in numeric_columns:
    plt.figure()
    sns.boxplot(x = df_test[col])
    plt.title(f"Boxplot {col}")
    plt.show()
    plt.close()

# analiza corelatiilor train
plt.figure(figsize = (10, 8))
correlation_mat = df_train[numeric_columns].corr()
sns.heatmap(correlation_mat, annot = True, cmap = 'coolwarm')
plt.title("Matricea de corelatii train")
plt.show()
plt.close()

# analiza corelatiilor test
plt.figure(figsize = (10, 8))
correlation_mat = df_test[numeric_columns].corr()
sns.heatmap(correlation_mat, annot = True, cmap = 'coolwarm')
plt.title("Matricea de corelatii test")
plt.show()
plt.close()

# analiza relatiilor cu variabila tinta pentru train
plt.figure()
sns.stripplot(x = 'bautura_pentru_tine', y = 'fericire',
            data=df_train, jitter = True)
plt.title("Fericire in functie de bautura")
plt.xticks(rotation = 45)
plt.show()
plt.close()

plt.figure()
sns.stripplot(x = 'bautura_pentru_tine', y = 'stres',
            data=df_train, jitter = True)
plt.title("Stres in functie de bautura")
plt.xticks(rotation = 45)
plt.show()
plt.close()

plt.figure()
sns.stripplot(x = 'bautura_pentru_tine', y = 'nr_ore_somn',
            data=df_train, jitter = True)
plt.title("Numar de ore de somn in functie de bautura")
plt.xticks(rotation = 45)
plt.show()
plt.close()

plt.figure()
sns.stripplot(x = 'bautura_pentru_tine', y = 'temperatura',
            data=df_train, jitter = True)
plt.title("Temperatura in functie de bautura")
plt.xticks(rotation = 45)
plt.show()
plt.close()

# analiza relatiilor cu variabila tinta pentru test
plt.figure()
sns.stripplot(x = 'bautura_pentru_tine', y = 'fericire',
            data=df_test, jitter = True)
plt.title("Fericire in functie de bautura")
plt.xticks(rotation = 45)
plt.show()
plt.close()

plt.figure()
sns.stripplot(x = 'bautura_pentru_tine', y = 'stres',
            data=df_test, jitter = True)
plt.title("Stres in functie de bautura")
plt.xticks(rotation = 45)
plt.show()
plt.close()

plt.figure()
sns.stripplot(x = 'bautura_pentru_tine', y = 'nr_ore_somn',
            data=df_test, jitter = True)
plt.title("Numar de ore de somn in functie de bautura")
plt.xticks(rotation = 45)
plt.show()
plt.close()

plt.figure()
sns.stripplot(x = 'bautura_pentru_tine', y = 'temperatura',
            data=df_test, jitter = True)
plt.title("Temperatura in functie de bautura")
plt.xticks(rotation = 45)
plt.show()
plt.close()

