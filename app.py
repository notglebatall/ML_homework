import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import pickle
import numpy as np
import re
from pathlib import Path

st.set_page_config(page_title="Предсказание цены автомобиля", page_icon="🚗", layout="wide")

MODEL_PATH = Path("model.pkl")
TRAIN_DATA_PATH = Path("df_train.csv")

@st.cache_resource
def load_model():
    with open(MODEL_PATH, 'rb') as f:
        artifacts = pickle.load(f)
    return artifacts

@st.cache_data
def load_train_data():
    return pd.read_csv(TRAIN_DATA_PATH, index_col=0)

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

try:
    df_train = load_train_data()
except Exception as e:
    st.error(f"❌ Ошибка загрузки тренировочных данных: {e}")
    st.stop()

st.title("🚗 Предсказание цены автомобиля")

tab1, tab2, tab3 = st.tabs(["EDA", "Рассчитать цену", "Веса модели"])


# EDA
with tab1:
    st.subheader("EDA")
    
    # Базовая информация о датасете
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Всего записей", f"{len(df_train):,}")
    with col2:
        st.metric("Признаков", len(df_train.columns))
    with col3:
        st.metric("Средняя цена", f"{df_train['selling_price'].mean():,.0f}")
    with col4:
        st.metric("Медианная цена", f"{df_train['selling_price'].median():,.0f}")
    
    st.write("**Описательная статистика:**")
    st.dataframe(df_train.describe())
    
    # Основные распределения
    st.subheader("📊 Основные распределения")
    col1, col2 = st.columns(2)
    
    with col1:
        # Распределение цен
        if 'selling_price' in df_train.columns:
            fig1 = px.histogram(df_train, x='selling_price', nbins=50, 
                               title="Распределение цен автомобилей")
            st.plotly_chart(fig1, use_container_width=True)
        
        # Распределение по годам
        if 'year' in df_train.columns:
            fig3 = px.histogram(df_train, x='year', nbins=30,
                               title="Распределение по годам выпуска")
            st.plotly_chart(fig3, use_container_width=True)
    
    with col2:
        # Цена vs год
        if 'year' in df_train.columns and 'selling_price' in df_train.columns:
            sample_size = min(500, len(df_train))
            fig2 = px.scatter(df_train.sample(sample_size), x='year', y='selling_price',
                             title="Зависимость цены от года выпуска",
                             trendline="lowess")
            st.plotly_chart(fig2, use_container_width=True)
        
        # Распределение по типу топлива
        if 'fuel' in df_train.columns:
            fuel_counts = df_train['fuel'].value_counts()
            fig4 = px.pie(values=fuel_counts.values, names=fuel_counts.index,
                         title="Распределение по типу топлива")
            st.plotly_chart(fig4, use_container_width=True)
    
    # Новый блок: Анализ категориальных признаков
    st.subheader("Анализ категориальных признаков")
    col1, col2 = st.columns(2)
    
    with col1:
        # Средняя цена по типу продавца
        if 'seller_type' in df_train.columns:
            avg_price_seller = df_train.groupby('seller_type')['selling_price'].mean().sort_values(ascending=False)
            fig_seller = px.bar(x=avg_price_seller.index, y=avg_price_seller.values,
                               title="Средняя цена по типу продавца",
                               labels={'x': 'Тип продавца', 'y': 'Средняя цена'})
            st.plotly_chart(fig_seller, use_container_width=True)
        
        # Распределение по владельцам
        if 'owner' in df_train.columns:
            owner_counts = df_train['owner'].value_counts()
            fig_owner = px.bar(x=owner_counts.index, y=owner_counts.values,
                              title="Распределение по количеству владельцев",
                              labels={'x': 'Владелец', 'y': 'Количество'})
            st.plotly_chart(fig_owner, use_container_width=True)
    
    with col2:
        # Средняя цена по типу коробки передач
        if 'transmission' in df_train.columns:
            avg_price_trans = df_train.groupby('transmission')['selling_price'].mean()
            fig_trans = px.bar(x=avg_price_trans.index, y=avg_price_trans.values,
                              title="Средняя цена по типу коробки передач",
                              labels={'x': 'Коробка передач', 'y': 'Средняя цена'})
            st.plotly_chart(fig_trans, use_container_width=True)
        
        # Boxplot цены по количеству мест
        if 'seats' in df_train.columns:
            fig_seats = px.box(df_train, x='seats', y='selling_price',
                              title="Распределение цены по количеству мест")
            st.plotly_chart(fig_seats, use_container_width=True)
    
    # Новый блок: Boxplot для выявления выбросов
    st.subheader("Анализ выбросов")
    numeric_cols = df_train.select_dtypes(include=[np.number]).columns
    numeric_cols = [col for col in numeric_cols if col != 'selling_price']
    
    selected_col = st.selectbox("Выберите признак для анализа выбросов:", numeric_cols)
    
    col1, col2 = st.columns(2)
    with col1:
        fig_box = px.box(df_train, y=selected_col, 
                        title=f"Boxplot для {selected_col}")
        st.plotly_chart(fig_box, use_container_width=True)
    
    with col2:
        Q1 = df_train[selected_col].quantile(0.25)
        Q3 = df_train[selected_col].quantile(0.75)
        IQR = Q3 - Q1
        outliers_count = ((df_train[selected_col] < Q1 - 1.5*IQR) | 
                         (df_train[selected_col] > Q3 + 1.5*IQR)).sum()
        
        st.metric("Количество выбросов", outliers_count)
        st.metric("Процент выбросов", f"{(outliers_count/len(df_train)*100):.2f}%")
        st.metric("Q1 (25%)", f"{Q1:.2f}")
        st.metric("Q3 (75%)", f"{Q3:.2f}")
        st.metric("IQR", f"{IQR:.2f}")
    
    # Корреляционная матрица
    st.subheader("Корреляционная матрица числовых признаков")
    numeric_cols_corr = df_train.select_dtypes(include=[np.number]).columns
    if len(numeric_cols_corr) > 0:
        corr_matrix = df_train[numeric_cols_corr].corr()
        
        fig5 = px.imshow(corr_matrix, text_auto='.2f', aspect="auto",
                        title="Корреляция признаков",
                        color_continuous_scale='RdBu_r')
        st.plotly_chart(fig5, use_container_width=True)
        
        # Топ корреляций с целевой переменной
        if 'selling_price' in corr_matrix.columns:
            st.write("**Топ-5 признаков по корреляции с ценой:**")
            price_corr = corr_matrix['selling_price'].drop('selling_price').abs().sort_values(ascending=False).head(5)
            fig_top_corr = px.bar(x=price_corr.values, y=price_corr.index, orientation='h',
                                 title="Топ-5 признаков по корреляции с ценой",
                                 labels={'x': 'Корреляция (по модулю)', 'y': 'Признак'})
            st.plotly_chart(fig_top_corr, use_container_width=True)
    else:
        st.warning("Нет числовых столбцов для построения корреляционной матрицы")
    
    # Новый блок: Анализ распределения таргета
    st.subheader("Анализ целевой переменной")
    col1, col2 = st.columns(2)
    
    with col1:
        fig_target = px.histogram(df_train, x='selling_price', nbins=50,
                                 title="Распределение цены (исходное)")
        st.plotly_chart(fig_target, use_container_width=True)
    
    with col2:
        fig_target_log = px.histogram(df_train, x=np.log1p(df_train['selling_price']), 
                                     nbins=50,
                                     title="Распределение log(цена)",
                                     labels={'x': 'log(selling_price)'})
        st.plotly_chart(fig_target_log, use_container_width=True)
    
    # Пропущенные значения
    st.subheader("Пропущенные значения")
    missing = df_train.isnull().sum()
    missing = missing[missing > 0].sort_values(ascending=False)
    if len(missing) > 0:
        fig6 = px.bar(x=missing.values, y=missing.index, orientation='h',
                     title="Количество пропущенных значений по столбцам")
        st.plotly_chart(fig6, use_container_width=True)
    else:
        st.success("✅ Пропущенных значений не обнаружено!")



# Предсказание цены
with tab2:
    st.subheader("Выберите способ ввода данных")
    
    input_method = st.radio(
        "Способ ввода:",
        ["Ручной ввод параметров", "Загрузка CSV файла"],
        horizontal=True
    )
    
    if input_method == "Ручной ввод параметров":
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
            
            submitted = st.form_submit_button("Предсказать цену", use_container_width=True)
        
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
                
                st.success(f"### 💰 Предсказанная цена: {predicted_price:,.0f}")
                
                # Показываем введенные данные
                with st.expander("Введенные данные"):
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
    
    else:  # Загрузка CSV файла
        st.subheader("Загрузите CSV файл с данными")
        st.info("⚠️ CSV файл должен содержать колонки: name, year, km_driven, fuel, seller_type, transmission, owner, mileage, engine, max_power, torque, seats")
        
        uploaded_file = st.file_uploader("Выберите CSV файл для предсказания", type=["csv"])
        
        if uploaded_file is not None:
            try:
                df_predict = pd.read_csv(uploaded_file, index_col=0)
                
                # Удаляем selling_price если он есть
                if 'selling_price' in df_predict.columns:
                    df_predict = df_predict.drop('selling_price', axis=1)
                
                # Проверяем и обрабатываем torque, если max_torque_rpm отсутствует
                if 'max_torque_rpm' not in df_predict.columns:
                    st.info("🔄 Обнаружен необработанный столбец torque. Выполняется парсинг...")
                    
                    def parse_torque(torque_str):
                        if pd.isna(torque_str) or torque_str == '':
                            return None, None

                        torque_str = str(torque_str).lower()

                        torque_value = None
                        torque_match = re.search(r'([\d.]+)\s*(?:nm|kgm)', torque_str)
                        if torque_match:
                            torque_value = float(torque_match.group(1))
                            if 'kgm' in torque_str:
                                torque_value = torque_value * 9.80665

                        rpm_value = None
                        rpm_match = re.search(r'[@at\s]+([\d,]+)(?:[-~]+([\d,]+))?\s*(?:\(?\s*(?:rpm|kgm)?)?', torque_str)
                        if rpm_match:
                            rpm1 = float(rpm_match.group(1).replace(',', ''))
                            if rpm_match.group(2):
                                rpm2 = float(rpm_match.group(2).replace(',', ''))
                                rpm_value = (rpm1 + rpm2) / 2
                            else:
                                rpm_value = rpm1

                        return torque_value, rpm_value
                    
                    df_predict[['torque', 'max_torque_rpm']] = df_predict['torque'].apply(
                        lambda x: pd.Series(parse_torque(x))
                    )
                    
                    # Заполняем пропуски медианами
                    if 'torque' in df_train.columns and 'max_torque_rpm' in df_train.columns:
                        torque_median = df_train['torque'].median()
                        rpm_median = df_train['max_torque_rpm'].median()
                        
                        df_predict['torque'].fillna(torque_median, inplace=True)
                        df_predict['max_torque_rpm'].fillna(rpm_median, inplace=True)
                    
                    st.success("✅ Парсинг torque завершен!")
                
                st.success(f"Файл успешно загружен! Строк: {len(df_predict)}, Столбцов: {len(df_predict.columns)}")
                
                with st.expander("👀 Просмотр загруженных данных"):
                    st.dataframe(df_predict.head(10))
                    st.write(f"**Размер данных:** {df_predict.shape}")
                    st.write(f"**Столбцы:** {', '.join(df_predict.columns.tolist())}")
                
                if st.button("Предсказать цены для всех строк", use_container_width=True):
                    try:
                        predictions = []
                        progress_bar = st.progress(0)
                        
                        for idx, (row_idx, row) in enumerate(df_predict.iterrows()):
                            # Извлекаем brand из name
                            brand = row['name'].split()[0] if pd.notna(row['name']) else 'Другое'
                            # Проверяем, является ли марка редкой
                            brand_counts = df_predict['name'].apply(lambda x: x.split()[0] if pd.notna(x) else 'Другое').value_counts()
                            if brand not in brand_counts.index or brand_counts[brand] < 20:
                                brand = 'Другое'
                            
                            # Создаем датафрейм с числовыми признаками
                            numeric_features = pd.DataFrame({
                                'year': [row['year']],
                                'km_driven': [row['km_driven']],
                                'mileage': [row['mileage']],
                                'engine': [row['engine']],
                                'max_power': [row['max_power']],
                                'torque': [row['torque']],
                                'seats': [row['seats']],
                                'max_torque_rpm': [row['max_torque_rpm']]
                            })

                            # Скейлим признаки
                            numeric_scaled = pd.DataFrame(
                                SCALER.transform(numeric_features),
                                columns=numeric_features.columns
                            )
                            
                            # Сохраняем seats для категориальных признаков
                            seats_value = int(numeric_features['seats'].iloc[0])
                            numeric_scaled = numeric_scaled.drop(columns='seats', axis=1)
                            
                            # Создаем датафрейм с категориальными признаками
                            categorical_features = pd.DataFrame({
                                'fuel': [row['fuel']],
                                'owner': [row['owner']],
                                'seats': [seats_value],
                                'seller_type': [row['seller_type']],
                                'transmission': [row['transmission']],
                                'brand': [brand]
                            })
                            
                            # OHE
                            categorical_encoded = pd.DataFrame(
                                OHE.transform(categorical_features),
                                columns=OHE.get_feature_names_out()
                            )
                            
                            # Feature Engineering
                            engineered_features = pd.DataFrame({
                                'year_squared': [row['year'] ** 2],
                                'power_per_liter': [row['max_power'] / row['engine']],
                                'torque_per_liter': [row['torque'] / row['engine']],
                                'specific_power': [row['max_power'] / (row['km_driven'] + 1)]
                            })
                            
                            engineered_scaled = pd.DataFrame(
                                NEW_SCALER.transform(engineered_features),
                                columns=engineered_features.columns
                            )
                            
                            # Feature Engineering номер 2
                            owner_third_or_more = float(row['owner'] in ['Third Owner', 'Fourth & Above Owner'])
                            premium_seller_first_owner = float(
                                (row['owner'] == 'First Owner') and (row['seller_type'] == 'Trustmark Dealer')
                            )
                            risk_combination = float(
                                (owner_third_or_more == 1) and (row['seller_type'] == 'Individual')
                            )
                            low_mileage_first_owner = float(
                                (row['owner'] == 'First Owner') and (row['km_driven'] < 50000)
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
                            predictions.append(predicted_price)
                            
                            # Обновляем прогресс-бар
                            progress_bar.progress((idx + 1) / len(df_predict))
                        
                        df_predict['predicted_price'] = predictions
                        
                        st.success(f"Расчеты выполнены для {len(predictions)} строк!")
                        
                        st.subheader("Результаты предсказаний")
                        st.dataframe(df_predict[['name', 'year', 'km_driven', 'predicted_price']])
                        
                        # Скачивание результатов
                        csv = df_predict.to_csv(index=True)
                        st.download_button(
                            label="📥 Скачать результаты в CSV",
                            data=csv,
                            file_name="predictions.csv",
                            mime="text/csv",
                            use_container_width=True
                        )
                        
                        # Статистика предсказаний
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Средняя цена", f"{df_predict['predicted_price'].mean():,.0f}")
                        with col2:
                            st.metric("Минимальная цена", f"{df_predict['predicted_price'].min():,.0f}")
                        with col3:
                            st.metric("Максимальная цена", f"{df_predict['predicted_price'].max():,.0f}")
                        
                    except Exception as e:
                        st.error(f"❌ Ошибка при предсказании: {e}")
                        st.write("Детали ошибки:")
                        st.exception(e)
                        
            except Exception as e:
                st.error(f"❌ Ошибка чтения файла: {e}")
                st.exception(e)
        else:
            st.info("Загрузите CSV файл для расчета цены")


# Визуализация весов модели
with tab3:
    st.subheader("Важность признаков модели")
    
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
        
        with st.expander("Все веса модели"):
            st.dataframe(weights_df)
        
        st.subheader("Метрики качества модели")
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