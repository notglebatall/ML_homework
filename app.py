import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import pickle
import numpy as np
from pathlib import Path

st.set_page_config(page_title="Предсказание цены автомобиля", page_icon="🚗", layout="wide")

MODEL_PATH = Path("model.pkl")

@st.cache_resource
def load_model():
    with open(MODEL_PATH, 'rb') as f:
        artifacts = pickle.load(f)
    return artifacts

try:
    artifacts = load_model()
    MODEL = artifacts['best_model']
    SCALER = artifacts['scaler']
    NEW_SCALER = artifacts['new_scaler']
    OHE = artifacts['ohe_encoder']
    FEATURE_NAMES = artifacts['feature_names']
except Exception as e:
    st.error(f"❌ Ошибка загрузки модели: {e}")
    st.stop()

st.title("🚗 Предсказание цены автомобиля")
st.subheader("📁 Загрузите CSV файл с данными")
uploaded_file = st.file_uploader("Выберите CSV файл для начала работы", type=["csv"])

if uploaded_file is None:
    st.info("👆 Пожалуйста, загрузите CSV файл для начала работы")
    st.stop()

# Загрузка файла
try:
    df = pd.read_csv(uploaded_file, index_col=0)
    st.success(f"✅ Файл успешно загружен! Строк: {len(df)}, Столбцов: {len(df.columns)}")
except Exception as e:
    st.error(f"❌ Ошибка чтения файла: {e}")
    st.stop()

# Просмотр первых строк
with st.expander("👀 Просмотр загруженных данных"):
    st.dataframe(df.head(10))
    st.write(f"**Размер данных:** {df.shape}")
    st.write(f"**Столбцы:** {', '.join(df.columns.tolist())}")

tab1, tab2, tab3 = st.tabs(["📊 EDA", "🔮 Рассчитать цену", "📈 Веса модели"])


# EDA
with tab1:
    st.subheader("📈 Exploratory Data Analysis")
    
    st.write("**Описательная статистика:**")
    st.dataframe(df.describe())
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Распределение цен
        if 'selling_price' in df.columns:
            fig1 = px.histogram(df, x='selling_price', nbins=50, 
                               title="Распределение цен автомобилей")
            st.plotly_chart(fig1, use_container_width=True)
        
        # Распределение по годам
        if 'year' in df.columns:
            fig3 = px.histogram(df, x='year', nbins=30,
                               title="Распределение по годам выпуска")
            st.plotly_chart(fig3, use_container_width=True)
    
    with col2:
        # Цена vs год
        if 'year' in df.columns and 'selling_price' in df.columns:
            sample_size = min(500, len(df))
            fig2 = px.scatter(df.sample(sample_size), x='year', y='selling_price',
                             title="Зависимость цены от года выпуска")
            st.plotly_chart(fig2, use_container_width=True)
        
        # Распределение по типу топлива
        if 'fuel' in df.columns:
            fuel_counts = df['fuel'].value_counts()
            fig4 = px.pie(values=fuel_counts.values, names=fuel_counts.index,
                         title="Распределение по типу топлива")
            st.plotly_chart(fig4, use_container_width=True)
    
    # Корреляционная матрица
    st.subheader("Корреляционная матрица числовых признаков")
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        corr_matrix = df[numeric_cols].corr()
        
        fig5 = px.imshow(corr_matrix, text_auto='.2f', aspect="auto",
                        title="Корреляция признаков",
                        color_continuous_scale='RdBu_r')
        st.plotly_chart(fig5, use_container_width=True)
    else:
        st.warning("Нет числовых столбцов для построения корреляционной матрицы")
    
    # Пропущенные значения
    st.subheader("Пропущенные значения")
    missing = df.isnull().sum()
    missing = missing[missing > 0].sort_values(ascending=False)
    if len(missing) > 0:
        fig6 = px.bar(x=missing.values, y=missing.index, orientation='h',
                     title="Количество пропущенных значений по столбцам")
        st.plotly_chart(fig6, use_container_width=True)
    else:
        st.success("✅ Пропущенных значений не обнаружено!")



# Предсказание цены
with tab2:
    st.subheader("✍️ Введите характеристики автомобиля")
    
    with st.form("prediction_form"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.write("**Основные характеристики:**")
            # Извлекаем марку из названия
            brand = st.selectbox("Марка автомобиля", 
                                ['Maruti', 'Hyundai', 'Honda', 'Tata', 'Mahindra', 
                                 'Ford', 'Renault', 'Chevrolet', 'Toyota', 'Другое'])
            year = st.number_input("Год выпуска", min_value=1990, max_value=2024, value=2015)
            km_driven = st.number_input("Пробег (км)", min_value=0, value=50000)
            fuel = st.selectbox("Тип топлива", ['Petrol', 'Diesel', 'CNG', 'LPG', 'Electric'])
            
        with col2:
            st.write("**Технические характеристики:**")
            engine = st.number_input("Объем двигателя (CC)", min_value=500, max_value=5000, value=1500)
            max_power = st.number_input("Мощность (bhp)", min_value=30.0, max_value=500.0, value=100.0)
            mileage = st.number_input("Расход топлива (kmpl)", min_value=5.0, max_value=40.0, value=18.0)
            torque = st.number_input("Крутящий момент (Nm)", min_value=50.0, max_value=500.0, value=190.0)
            max_torque_rpm = st.number_input("Обороты макс. крутящего момента (rpm)", 
                                            min_value=1000, max_value=6000, value=2000)
            
        with col3:
            st.write("**Дополнительно:**")
            seats = st.selectbox("Количество мест", [2, 4, 5, 6, 7, 8, 9], index=2)
            transmission = st.selectbox("Коробка передач", ['Manual', 'Automatic'])
            owner = st.selectbox("Владелец", ['First Owner', 'Second Owner', 'Third Owner', 
                                              'Fourth & Above Owner'])
            seller_type = st.selectbox("Тип продавца", ['Individual', 'Dealer', 'Trustmark Dealer'])
        
        submitted = st.form_submit_button("🔮 Предсказать цену", use_container_width=True)
    
    if submitted:
        try:
            # Создаем датафрейм с числовыми признаками
            numeric_features = pd.DataFrame({
                'year': [year],
                'km_driven': [km_driven],
                'mileage': [mileage],
                'engine': [engine],
                'max_power': [max_power],
                'torque': [torque],
                'seats': [seats],
                'max_torque_rpm': [max_torque_rpm]

            })

            # Скейлим признаки
            numeric_scaled = pd.DataFrame(
                SCALER.transform(numeric_features),
                columns=numeric_features.columns
            )

            numeric_scaled = numeric_scaled.drop(columns='seats', axis=1)
            
            # Создаем датафрейм с категориальными признаками
            categorical_features = pd.DataFrame({
                'fuel': [fuel],
                'owner': [owner],
                'seats': [seats],
                'seller_type': [seller_type],
                'transmission': [transmission],
                'brand': [brand]
            })

            
            # OHE
            categorical_encoded = pd.DataFrame(
                OHE.transform(categorical_features),
                columns=OHE.get_feature_names_out()
            )
            

            
            # Feature Engineering
            engineered_features = pd.DataFrame({
                'year_squared': [year ** 2],
                'power_per_liter': [max_power / engine],
                'torque_per_liter': [torque / engine],
                'specific_power': [max_power / (km_driven + 1)]
            })
            
            engineered_scaled = pd.DataFrame(
                NEW_SCALER.transform(engineered_features),
                columns=engineered_features.columns
            )
            
            # Feature Engineering номер 2
            owner_third_or_more = float(owner in ['Third Owner', 'Fourth & Above Owner'])
            premium_seller_first_owner = float(
                (owner == 'First Owner') and (seller_type == 'Trustmark Dealer')
            )
            risk_combination = float(
                (owner_third_or_more == 1) and (seller_type == 'Individual')
            )
            low_mileage_first_owner = float(
                (owner == 'First Owner') and (km_driven < 50000)
            )
            
            binary_features = pd.DataFrame({
                'owner_third_or_more': [owner_third_or_more],
                'premium_seller_first_owner': [premium_seller_first_owner],
                'risk_combination': [risk_combination],
                'low_mileage_first_owner': [low_mileage_first_owner]
            })
            
            # Объединяем все в один датафрейм
            X_final = pd.concat([
                numeric_scaled,
                categorical_encoded,
                engineered_scaled,
                binary_features
            ], axis=1)
            
            # Убеждаемся, что все признаки на месте
            missing_features = set(FEATURE_NAMES) - set(X_final.columns)
            for feature in missing_features:
                X_final[feature] = 0
            
            X_final = X_final[FEATURE_NAMES]
            
            # Делаем предсказание
            predicted_price = MODEL.predict(X_final)[0]
            
            st.success(f"### 💰 Предсказанная цена: ₹{predicted_price:,.0f}")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Минимальная оценка", f"₹{int(predicted_price * 0.9):,}")
            with col2:
                st.metric("Средняя оценка", f"₹{int(predicted_price):,}")
            with col3:
                st.metric("Максимальная оценка", f"₹{int(predicted_price * 1.1):,}")
            
            # Показываем введенные данные
            with st.expander("📋 Введенные данные"):
                input_data = {
                    'Марка': brand,
                    'Год': year,
                    'Пробег': f"{km_driven:,} км",
                    'Топливо': fuel,
                    'Двигатель': f"{engine} CC",
                    'Мощность': f"{max_power} bhp",
                    'Расход': f"{mileage} kmpl",
                    'Крутящий момент': f"{torque} Nm @ {max_torque_rpm} rpm",
                    'Места': seats,
                    'КПП': transmission,
                    'Владелец': owner,
                    'Продавец': seller_type
                }
                st.json(input_data)
                
        except Exception as e:
            st.error(f"❌ Ошибка при предсказании: {e}")
            st.write("Детали ошибки:")
            st.exception(e)


# Визуализация весов модели
with tab3:
    st.subheader("📊 Важность признаков модели")
    
    try:
        coefficients = MODEL.coef_
        
        weights_df = pd.DataFrame({
            'Feature': FEATURE_NAMES,
            'Weight': coefficients,
            'Abs_Weight': np.abs(coefficients)
        }).sort_values('Abs_Weight', ascending=False)
        
        top_features = weights_df.head(15)
        
        fig = px.bar(top_features, x='Weight', y='Feature', orientation='h',
                     title="Топ-15 самых важных признаков",
                     color='Weight',
                     color_continuous_scale='RdBu_r')
        st.plotly_chart(fig, use_container_width=True)
        
        with st.expander("📋 Все веса модели"):
            st.dataframe(weights_df)
        
        st.subheader("📈 Метрики качества модели")
        metrics = artifacts['model_metrics']
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Модель", metrics['best_model_name'])
        with col2:
            st.metric("R² Score", f"{metrics['test_r2']:.4f}")
        with col3:
            st.metric("MSE", f"{metrics['test_mse']:,.0f}")
        with col4:
            st.metric("Business Metric", f"{metrics['business_metric']:.2%}")
        
        st.subheader("Распределение весов")
        fig_dist = px.histogram(weights_df, x='Weight', nbins=50,
                               title="Распределение весов признаков")
        st.plotly_chart(fig_dist, use_container_width=True)
        
    except Exception as e:
        st.error(f"Ошибка визуализации весов: {e}")
