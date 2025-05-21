import pandas as pd
import numpy as np

# generez date sintetice pentru a vedea ce bautura se potriveste
# unui consuator in functie de preferinte si evenimente
def genereaza_date_bauturi(nr_samples = 840):
    def random_ore_somn():
        return np.random.randint(4, 12)
    def random_preferinta_zahar():
        return np.random.choice(['putin', 'mult'])
    def random_preferinta_acrime():
        return np.random.choice(['putina', 'multa'])
    def random_alcool():
        return np.random.choice(['da', 'nu'])
    def random_grad_de_fericire():
        return round(np.random.uniform(0, 10), 2)
    def random_grad_de_stres():
        return round(np.random.uniform(0, 10), 2)
    def random_vreme():
        return np.random.choice(['insorit', 'racoros', 'ploios'])
    def random_temperatura():
        return np.random.randint(-10, 30)
    def random_energie():
        return np.random.choice(['putina', 'multa'])

    # algoritm de potrivire a bauturii
    def primeste_bautura(row):
        if (row['nr_ore_somn'] < 7 and row['multa_treaba'] == 'da' and
            row['energie'] == 'putina'):
            return 'cafea'
        if (row['vreme'] == 'racoros' and row['alcool'] == 'nu' and
            row['temperatura'] < 10):
            return 'ciocolata calda'
        if (row['vreme'] == 'ploios' and row['alcool'] == 'nu' and
            row['stres'] > 3):
            return 'ceai'
        if row['fericire'] < 3 and row['alcool'] == 'da':
            return 'bere'
        if (row['fericire'] > 7 and row['alcool'] == 'da' and
            row['energie'] == 'multa'):
            return 'vin'
        if row['cat_de_acru'] == 'multa' and row['alcool'] == 'da':
            return 'cocktail'
        if (row['cat_de_acru'] == 'multa' and row['alcool'] == 'nu' and
            row['vreme'] == 'insorit'):
            return 'limonada'
        if row['cantitate_zahar'] == 'mult' and row['alcool'] == 'nu':
            return 'suc'
        if (row['cantitate_zahar'] == 'putin' and row['alcool'] == 'nu' and
            row['fericire'] > 5):
            return 'compot'
        if row['temperatura'] > 15 and row['energie'] == 'putina':
            return 'apa'
        if row['alcool'] == 'da':
            return 'cocktail'
        return 'apa'

    # generez datele intr-un DataFrame
    df = pd.DataFrame({
        'nr_ore_somn': [random_ore_somn() for i in range(nr_samples)],
        'multa_treaba': [np.random.choice(['da', 'nu']) for i in range(nr_samples)],
        'energie': [random_energie() for i in range(nr_samples)],
        'fericire': [random_grad_de_fericire() for i in range(nr_samples)],
        'alcool': [random_alcool() for i in range(nr_samples)],
        'cat_de_acru': [random_preferinta_acrime() for i in range(nr_samples)],
        'cantitate_zahar': [random_preferinta_zahar() for i in range(nr_samples)],
        'vreme': [random_vreme() for i in range(nr_samples)],
        'temperatura': [random_temperatura() for i in range(nr_samples)],
        'stres': [random_grad_de_stres() for i in range(nr_samples)]
    })

    # aplic functia pe fiecare rand pentru a determina bautura
    df['bautura_pentru_tine'] = df.apply(primeste_bautura, axis=1)

    from sklearn.model_selection import train_test_split

    # X nu contine coloana tinta
    X = df.drop(columns=['bautura_pentru_tine'])
    # y contine doar coloana tinta
    y = df['bautura_pentru_tine']
    # impart datele in seturi de antrenare si testare
    X_train, X_test, y_train, y_test = train_test_split(X, y,
        test_size=0.286, random_state=42)

    train_df = X_train.copy()
    train_df['bautura_pentru_tine'] = y_train
    test_df = X_test.copy()
    test_df['bautura_pentru_tine'] = y_test

    train_df.to_csv('train111.csv', index = False)
    test_df.to_csv('test111.csv', index = False)

genereaza_date_bauturi()