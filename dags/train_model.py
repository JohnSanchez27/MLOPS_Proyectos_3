from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import recall_score, classification_report

def train_model():
    # Obtener los datos ya preprocesados
    df_train, df_val, df_test = preprocesar_datos()

    # Separar features (X) y etiquetas (y)
    X_train = df_train.drop("readmitted", axis=1)
    y_train = df_train["readmitted"]

    X_val = df_val.drop("readmitted", axis=1)
    y_val = df_val["readmitted"]

    X_test = df_test.drop("readmitted", axis=1)
    y_test = df_test["readmitted"]

    # Identificar variables categóricas y numéricas
    categorical_features = X_train.select_dtypes(include="object").columns.tolist()
    numeric_features = X_train.select_dtypes(include=np.number).columns.tolist()

    print("Variables categóricas:", categorical_features)
    print("Variables numéricas:", numeric_features)

    # Definir el modelo y pipeline
    model = RandomForestClassifier(random_state=42)

    preprocessor = ColumnTransformer(transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(sparse_output=False, handle_unknown='ignore'), categorical_features)
    ])

    pipe = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', model)
    ])

    # Entrenar el modelo
    pipe.fit(X_train, y_train)

    # Evaluar en el conjunto de validación
    score = pipe.score(X_val, y_val)
    print(f"Accuracy en validación: {score:.4f}")

    return pipe, X_test, y_test
