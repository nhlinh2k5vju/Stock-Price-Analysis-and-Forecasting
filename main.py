import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping

file_path = "./data/processed/data.xlsx"
stock_price_df = pd.read_excel(file_path, sheet_name='stock_price')
stock_price_df['date'] = pd.to_datetime(stock_price_df['date'])

# Mã cổ phiếu cần dự đoán
companies = ['HPG', 'HSG']

# Tiền xử lý dữ liệu
def preprocess_data(data, company):
    print(f"\nPreprocessing data for {company}...")
    data = data[data['stock_name'] == company].sort_values('date')
    print(f"Initial stats:\n{data.describe()}\n")
    
    # Kiểm tra dữ liệu thiếu
    if data.isnull().values.any():
        print(f"Missing values detected for {company}, filling with forward fill.")
        data = data.fillna(method='ffill')
    
    # Kiểm tra outliers (ngoại lệ)
    Q1 = data['close'].quantile(0.25)
    Q3 = data['close'].quantile(0.75)
    IQR = Q3 - Q1
    outlier_condition = (data['close'] < (Q1 - 1.5 * IQR)) | (data['close'] > (Q3 + 1.5 * IQR))
    if outlier_condition.any():
        print(f"Outliers detected for {company}, consider handling.")
    
    return data

# Duyệt qua từng mã cổ phiếu
for company in companies:

    company_data = preprocess_data(stock_price_df, company)
    close_data = company_data[['date', 'close']].drop(columns=['date'])

    # Chuẩn hóa dữ liệu
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(close_data.values)

    # Chia dữ liệu thành tập huấn luyện và tập kiểm tra
    training_size = int(np.ceil(len(scaled_data) * 0.95))
    train_data = scaled_data[:training_size]
    test_data = scaled_data[training_size - 60:]  # Bao gồm 60 ngày cuối cho kiểm tra

    # Chuẩn bị dữ liệu huấn luyện
    x_train, y_train = [], []
    for i in range(60, len(train_data)):
        x_train.append(train_data[i-60:i, 0])
        y_train.append(train_data[i, 0])

    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    # Xây dựng mô hình LSTM
    model = keras.Sequential([
        keras.layers.LSTM(units=128, return_sequences=True, input_shape=(x_train.shape[1], 1)),
        keras.layers.LSTM(units=128),
        keras.layers.Dense(32),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(1)
    ])

    model.compile(optimizer='adam', loss='mean_squared_error')

    # Tạo callback EarlyStopping
    early_stopping = EarlyStopping(monitor='val_loss', patience=20, verbose=1, restore_best_weights=True)

    # Huấn luyện mô hình với EarlyStopping
    history = model.fit(
        x_train, y_train,
        epochs=200, batch_size=32,
        validation_split=0.1,
        callbacks=[early_stopping]
    )

    # Vẽ biểu đồ loss và val_loss
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title(f'Training and Validation Loss for {company}')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    # Chuẩn bị dữ liệu kiểm tra
    x_test = []
    y_test = close_data.values[training_size:]

    for i in range(60, len(test_data)):
        x_test.append(test_data[i-60:i, 0])

    x_test = np.array(x_test)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    # Dự đoán dữ liệu kiểm tra
    predictions = model.predict(x_test)
    predictions = scaler.inverse_transform(predictions)

    # Tính toán MSE và RMSE
    mse = np.mean((predictions - y_test) ** 2)
    rmse = np.sqrt(mse)
    print(f"{company} MSE: {mse}, RMSE: {rmse}")

    # Vẽ đồ thị
    train = company_data[:training_size]
    test = company_data[training_size:]
    test['Predictions'] = predictions

    plt.figure(figsize=(10, 8))
    plt.plot(train['date'], train['close'], label='Train')
    plt.plot(test['date'], test['close'], label='Actual')
    plt.plot(test['date'], test['Predictions'], label='Predictions', linestyle='dashed')
    plt.title(f'{company} Stock Price Prediction')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.show()

    # Dự báo thêm 15 ngày
    last_60_days = test_data[-60:]  # Dữ liệu của 60 ngày cuối cùng từ tập kiểm tra
    last_60_days = np.reshape(last_60_days, (1, last_60_days.shape[0], 1))  # Chuyển dạng cho LSTM

    # Dự đoán 15 ngày tiếp theo
    forecast_days = 15
    forecast_prices = []

    for _ in range(forecast_days):
        pred = model.predict(last_60_days)
        forecast_prices.append(pred[0, 0])
        last_60_days = np.append(last_60_days[:, 1:, :], pred.reshape(1, 1, 1), axis=1)

    # Đảo ngược chuẩn hóa dữ liệu dự báo
    forecast_prices = scaler.inverse_transform(np.array(forecast_prices).reshape(-1, 1))

    # Tạo mảng ngày cho các dự báo tương lai
    last_date = test['date'].iloc[-1]
    forecast_dates = [last_date + pd.Timedelta(days=i) for i in range(1, forecast_days + 1)]

    # Vẽ đồ thị với dự báo 15 ngày
    plt.figure(figsize=(10, 8))
    plt.plot(train['date'], train['close'], label='Train')
    plt.plot(test['date'], test['close'], label='Actual')
    plt.plot(test['date'], test['Predictions'], label='Predictions', linestyle='dashed')
    plt.plot(forecast_dates, forecast_prices, label='15-Day Forecast', linestyle='dotted', color='red')
    plt.title(f'{company} Stock Price Prediction with 15-Day Forecast')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.show()
