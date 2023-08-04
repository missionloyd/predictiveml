import numpy as np
import matplotlib.pyplot as plt

def plot_results(y_test, y_pred, nan_mask, bldgname, y_column, datelevel, model_file):        
    # plot results
    fig, ax = plt.subplots()

    # Plot the actual values
    ax.plot(y_test, label='Actual Values', alpha=0.7)

    # Plot the predictions
    ax.plot(y_pred, label='Forecasted Values', alpha=0.8)

    # Plot the replaced missing values
    y_test[~nan_mask] = np.nan

    ax.plot(y_test, label='Predicted Values', alpha=0.75)

    ax.set_title(f'{bldgname} Consumption')
    ax.set_xlabel(f'Time ({datelevel}s)')
    ax.set_ylabel(y_column.split('_')[-2] + ' (' + y_column.split('_')[-1] + ')')

    ax.legend()
    plt.grid(True)
    plt.savefig(model_file + '.png')
    plt.close(fig) 
    
    return