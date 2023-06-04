from prophet import Prophet

def prophet(model_data):
    m = Prophet()
    m.fit(model_data)
    future = m.make_future_dataframe(periods=0, freq='H')
    forecast = m.predict(future)
    model_data['y'] = model_data['y'].fillna(forecast['yhat'])
    return model_data