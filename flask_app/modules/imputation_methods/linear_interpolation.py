def linear_interpolation(model_data):
    model_data['y'] = model_data['y'].interpolate(method='linear', limit_direction='both') 
    return model_data