# WineQualityPredictor

Este proyecto es una aplicación web construida con **Django** que predice la calidad del vino utilizando un modelo de machine learning. La aplicación incluye un backend en **Django** y puede ser desplegada en Heroku.

## Características principales

- Predicción de la calidad del vino a partir de sus características químicas.
- API de predicción lista para integrarse con un frontend (por ejemplo, React).
- Desplegable en Heroku.

---

## Instalación y configuración

### 1. Clonar el repositorio

```bash
git clone https://github.com/HernanVilca/BeDjangoPredictorWine.git
cd BeDjangoPredictorWine
```

### 2. Crear y activar un entorno virtual

```bash
python -m venv venv
.\venv\Scripts\activate
```

### 3. Instalar las dependencias

```bash
pip install -r requirements.txt

```

### 4. Migrar la base de datos

```bash
python manage.py migrate


```

### 5. Ejecutar el servidor de desarrollo

```bash
python manage.py runserver


```

### 6. Uso de la API

```bash
http://127.0.0.1:8000/predictorWine/predict/?fixed_acidity=7.4&volatile_acidity=0.7&citric_acid=0.0&residual_sugar=1.9&chlorides=0.076&free_sulfur_dioxide=11.0&total_sulfur_dioxide=34.0&density=0.9978&pH=3.51&sulphates=0.56&alcohol=9.4


```
