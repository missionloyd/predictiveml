def zero_fill(model_data):
    model_data['y'] = model_data['y'].fillna(0) 
    return model_data