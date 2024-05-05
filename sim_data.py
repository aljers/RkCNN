import pandas as pd
import numpy as np
"""
In the simulation data set, we have generated two classes.
One with 20 informative features are generated from a (correlated) 
multi-variate normal(2, sigma) distribution, the other one with 20 informative 
features are generated from independent normal(1,1) distribution.
User could tune the dataset by adding non-informative features (noises)

parameters:
variable_size: 
num_variables: the number of (informative) features added into dataset
num_noises: default = 0, the number of non-informative features added into dataset
"""
def get_data(variable_size = None, num_variables = None, num_noises = 0, omega = 1):
    # Mean vector
    mean = np.zeros(num_variables) + 2

    # Covariance matrix
    sig = np.zeros((num_variables,num_variables))
    for i in range(num_variables):
        for j in range(num_variables):
            sig[i][j] = (0.5)**np.abs(i-j)

    # Cholesky decomposition
    L = np.linalg.cholesky(omega * sig)

    # Generate independent standard normal variables
    std_normal_vars = np.random.normal(0, 1, size=(variable_size, num_variables))

    # Transform variables using the Cholesky decomposition
    correlated_vars = mean + np.dot(std_normal_vars, L.T)

    # Generate variables of size 500 from a standard normal distribution
    variables = np.zeros((num_variables, variable_size))
    for i in range(num_variables):
        variables[i] = np.random.normal(1, 1, size=variable_size)

    data_0 = pd.DataFrame(correlated_vars, columns=['feature_'+str(i+1) for i in range(num_variables)])
    data_1 = pd.DataFrame(variables.T, columns=['feature_'+str(i+1) for i in range(num_variables)])
    res_0 = pd.Series(np.zeros(variable_size))
    res_1 = pd.Series(np.zeros(variable_size) + 1)
    informative = pd.concat([data_0, data_1], ignore_index=True)
    response = pd.concat([res_0, res_1], ignore_index=True)
    response = pd.DataFrame(response)
    size = (2*variable_size, num_noises)

    # Create random noises
    noises = pd.DataFrame(np.random.randn(*size), columns=['noises_'+str(i+1) for i in range(num_noises)])
    final = informative.join(noises)

    return final,response

def get_data3(variable_size = None, num_variables = None, num_noises = 0, omega = 1, omegaC = 3):
    # Mean vector
    mean = np.zeros(num_variables) + 2

    # Covariance matrix
    sig = np.zeros((num_variables, num_variables))
    for i in range(num_variables):
        for j in range(num_variables):
            sig[i][j] = (0.5)**np.abs(i-j)

    # Cholesky decomposition
    L = np.linalg.cholesky(omega * sig)

    # Generate independent standard normal variables
    std_normal_vars = np.random.normal(0, 1,
                                       size=(variable_size, num_variables))

    # Transform variables using the Cholesky decomposition
    correlated_vars = mean + np.dot(std_normal_vars, L.T)

    # Generate variables of size 500 from a standard normal distribution
    variables = np.zeros((num_variables, variable_size))
    for i in range(num_variables):
        variables[i] = np.random.normal(1, 1,
                                        size=variable_size)

    variables2 = np.zeros((num_variables, variable_size))
    for i in range(num_variables):
        variables2[i] = np.random.normal(3, np.sqrt(omegaC),
                                         size=variable_size)

    data_0 = pd.DataFrame(correlated_vars,
                          columns=['feature_'+str(i+1) for i in range(num_variables)])
    data_1 = pd.DataFrame(variables.T,
                          columns=['feature_'+str(i+1) for i in range(num_variables)])
    data_2 = pd.DataFrame(variables2.T,
                          columns=['feature_'+str(i+1) for i in range(num_variables)])
    res_0 = pd.Series(np.zeros(variable_size))
    res_1 = pd.Series(np.zeros(variable_size) + 1)
    res_2 = pd.Series(np.zeros(variable_size) + 2)
    informative = pd.concat([data_0, data_1, data_2], ignore_index=True)
    response = pd.concat([res_0, res_1, res_2], ignore_index=True)
    response = pd.DataFrame(response)
    size = (3*variable_size, num_noises)

    # Create random noises
    noises = pd.DataFrame(np.random.randn(*size),
                          columns=['noises_'+str(i+1) for i in range(num_noises)])
    final = informative.join(noises)

    return final, response