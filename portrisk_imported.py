import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


path = "nasdaq_data"
files = Path(path).glob('*.csv')
df_list = list()
for file in list(files): 
    data = pd.read_csv(file)
    data['stock'] = file.stem
    if (len(data) > 1 and data['date'].iloc[0] == "2018-11-02"):
        df_list.append(data.loc[(pd.to_datetime(data['date']) > "2014-01-02" ) & (data['date'] <= "2018-11-02")])
       
df = pd.concat(df_list, ignore_index = True)

def generate_weight(n_assets):
    weight = np.random.random(n_assets)
    weight /= weight.sum()
    return weight

def generate_portfolio(df):
    return df.pivot(index = 'date', columns = 'stock', values = 'adjclose').pct_change()

def trim_portfolio(portfolio):
    trimmed_portfolio = portfolio.iloc[1:].dropna(axis = 1, thresh = 200)
    portfolio_mean = trimmed_portfolio.dropna(axis = 1).mean(axis = 0)
    portfolio_mean_sorted = portfolio_mean.sort_values()
    return (trimmed_portfolio.drop(portfolio_mean_sorted.tail(20).index, axis = 1), 
           portfolio_mean.drop(portfolio_mean_sorted.tail(20).index))

def calculate_portfolio_volatility(weight, covariance):
    return np.sqrt(np.dot(weight.T, np.dot(weight, covariance * 252)))


def display_assets(portfolio, portfolio_mean):
    asset_returns = portfolio_mean.to_numpy() * 252
    asset_volatility = list()
    for col in portfolio.columns.values:
        asset_volatility.append(np.sqrt(np.var(portfolio[col]) * 252))
    asset_volatility = np.array(asset_volatility)
    returns_vs_vol = pd.DataFrame({'Return': asset_returns, 'Volatility': asset_volatility})
    fig = returns_vs_vol.plot.scatter(x='Volatility', y = 'Return', figsize= (10, 10))
    plt.xlabel('Volatility')
    plt.ylabel('Return')
    return fig

def optimize_portfolio(num_portfolios, portfolio_mean, portfolio):
    port_returns = list()
    port_vols = list()
    weights = list()
    for x in range(num_portfolios):
        class Data:
            def __init__(self):
                self.weights = generate_weight(len(portfolio.columns))
                self.returns = (calculate_expected_return(portfolio_mean, self.weights) * 252)
                self.volatility = (calculate_portfolio_volatility(self.weights, portfolio.cov().to_numpy()))
        data = Data()
        port_returns.append(data.returns)
        port_vols.append(data.volatility)
        weights.append(data.weights)
    port_returns = np.array(port_returns)
    port_vols = np.array(port_vols)
    weights = np.array(weights)
    portfolios = pd.DataFrame({'Return': port_returns, 'Volatility': port_vols})
    fig = portfolios.plot.scatter(x='Volatility', y = 'Return', figsize= (10, 10))
    plt.xlabel('Expected Volatility')
    plt.ylabel('Expected Return')
    optimal_portfolio = weights[(portfolios['Return']/portfolios['Volatility']).idxmax()]
    return fig, optimal_portfolio


def calculate_weight(df, portfolio):
    df_close = df.pivot(index = 'date', columns = 'stock', values = 'close')
    weights = df_close[portfolio.columns].iloc[-1]
    portfolio_value = weights.sum()
    weights = weights / portfolio_value
    return weights.to_numpy()

def calculate_expected_return(portfolio_means, weight):
   return np.dot(weight, portfolio_means)

def calculate_risk(portfolio, weight):
    cov_matrix = portfolio.cov().to_numpy()
    return np.sqrt(weight.T @ cov_matrix @ weight)

def generate_components(df, n_components):
    pca = PCA(n_components= n_components)
    pca.fit(df.transpose())
    return pd.DataFrame(pca.components_).transpose()

def generate_factor_weights_and_r2(factors, returns):
    regr = LinearRegression()
    weights = list()
    r2s = list()
    intercepts = list()
    for col in returns.columns.values:
        y = returns.loc[:, col].values.reshape(-1, 1)
        regr.fit(factors, y) 
        r2 = r2_score(y, regr.predict(factors))
        weights.append(np.reshape(regr.coef_, -1))
        r2s.append(r2)
        intercepts.append(regr.intercept_)
    return np.array(weights), np.array(r2s), np.array(intercepts)

def calculate_specific_risk(returns, r2s):
    specific_risks = list()
    for col, r2 in zip(returns.columns.values, r2s):
        var = np.var(returns.loc[:, col].values.reshape(-1, 1))
        specific_risk = (r2 - 1) * -var
        specific_risks.append(specific_risk)
    return np.array(specific_risks)


def calculate_factor_risk_and_returns(components, weights, specific_risk, factor_weights, intercepts):
    bp = weights.T @ factor_weights
    return (np.sqrt((bp @ (components.cov() @ bp.T)) + weights.T @ (np.diag(specific_risk) @ weights)), 
            bp @ components.mean(axis = 0) + (weights.T @ intercepts).item())

def calculate_position_risk_attribution(df, weights, variance):
    marginal_risk = df.cov().to_numpy() @ weights.T
    attributions = np.multiply(marginal_risk, weights.T)/variance
    return attributions, pd.DataFrame({'Stocks': df.columns.values, 'Attribution': attributions}).sort_values(by = 'Attribution')

def calculate_factor_risk_attribution(components, weights, specific_risk, factor_weights, variance):
    bp2 = factor_weights.T @ weights
    marginal_risk = factor_weights @ (components.cov().to_numpy() @ bp2)
    specific_risk_contribution = specific_risk @ weights
    attributions = (np.multiply(factor_weights.T, (marginal_risk + specific_risk_contribution)))/variance
    attributions_df = pd.DataFrame({'Attributions': attributions})
    return attributions, attributions_df.sort_values(by= 'Attributions')

def find_riskiest_stocks(attributions, n_stocks):
    return (attributions.tail(n_stocks)).to_numpy()

def find_riskiest_factors(attributions, n_factors):
    return (attributions.tail(n_factors))
class Helper_Info:
    def __init__(self, df):
        self.components = generate_components(df, 20)
        self.factor_weights, self.r2, self.intercepts = generate_factor_weights_and_r2(self.components, df)
        self.specific_risk = calculate_specific_risk(df, self.r2)

portfolio = generate_portfolio(df)
trimmed_portfolio, trimmed_portfolio_means = trim_portfolio(portfolio)
trimmed_portfolio = trimmed_portfolio.dropna(axis=1)


#asset_plot = display_assets(trimmed_portfolio, trimmed_portfolio_means)
plot, weights = optimize_portfolio(1000, trimmed_portfolio_means, trimmed_portfolio)
#plt.show()
#weights_unoptimized = calculate_weight(df, trimmed_portfolio)
risk = calculate_risk(trimmed_portfolio, weights)
returns = calculate_expected_return(trimmed_portfolio_means, weights)
helper_info = Helper_Info(trimmed_portfolio)
factor_risk, factor_return = calculate_factor_risk_and_returns(helper_info.components, weights, helper_info.specific_risk, helper_info.factor_weights, helper_info.intercepts)
position_level_risk_attribution, position_level_risk_attribution_dataframe = calculate_position_risk_attribution(trimmed_portfolio, weights, risk)
factor_risk_attribution, factor_risk_attribution_dataframe = calculate_factor_risk_attribution(helper_info.components, weights, helper_info.specific_risk, helper_info.factor_weights, factor_risk)
print(factor_risk_attribution_dataframe)
riskiest_ten_stocks = find_riskiest_stocks(position_level_risk_attribution_dataframe, 10)
print(riskiest_ten_stocks)
riskiest_ten_factors = find_riskiest_factors(factor_risk_attribution_dataframe, 10)
print(riskiest_ten_factors)
print("portfolio analysis using optimal weights obtained through efficient frontier")
print("non factor risk: ", risk)
print("position based attribution sum :", position_level_risk_attribution.sum())
print("factor based risk: ", factor_risk)
print("factor based attribution sum: ", factor_risk_attribution.sum())
print("non factor return: ", returns)
print("facotr based return:", factor_return)