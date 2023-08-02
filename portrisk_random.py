import numpy as np


def generate_weight(n_assets):
    weight = np.random.random(n_assets)
    weight /= weight.sum()
    return weight

def generate_portfolio(years, assets, days=252):
    portfolio = np.empty((years * days, assets))
    for i in range(assets):
        portfolio[:, i] = np.random.uniform(low = -0.03, high = 0.05, size = years * days)
    return portfolio

def calculate_expected_return(portfolio, weight):
    return_vector = np.mean(portfolio, axis = 0)
    return np.dot(weight, return_vector)

def calculate_risk(portfolio, weight):
    cov_matrix = np.cov(portfolio, rowvar = False)
    return np.sqrt(np.matmul(weight, np.matmul(cov_matrix, np.transpose(weight))))

portfolio = generate_portfolio(4, 1500)
weights = generate_weight(np.size(portfolio, 1))

print(calculate_expected_return(portfolio, weights) * 252)
print(calculate_risk(portfolio, weights) * np.sqrt(252))

