"""
Todo:
Description
Available objective function are given as obj_*()
Available constraints are given as cons_*()
"""
import numpy as np
from numpy.linalg import norm
import pandas as pd
from scipy.optimize import minimize
from . import test_result

# risk budgeting approach optimisation object function


def obj_RC(w, p_cov):
    """Objective function for minimize RC. Given portfolio w and covariance matrix p_cov, return the variance measurement RC(w).

    Args:
            w : The portfolio of N assets with shape (N, ).
            p_cov : The covariance matrix of assets with shape (N x N).

    Returns:
            The variance measurement RC(w).
    """
    return norm(w*p_cov@w)**2/(w.T@p_cov@w)


def obj_Exp(w, mu):
    """Objective function for maximize expected return of the portfolio w.

    Args:
            w : The portfolio of N assets with shape (N, ).
            mu : The expected return of each asset with shape (N, ).

    Returns:
            The minus of expected return of the portfolio w. Note that the minus is due to min -f(x) <=> max f(x).
    """
    return - w.T@mu


def obj_Exp_minus_RC(w, mu, p_cov):
    """Objective function for maximize expected return of the portfolio w minus the variance measurement RC(w).

    Args:
            w : The portfolio of N assets with shape (N, ).
            mu : The expected return of each asset with shape (N, ).
            p_cov : The covariance matrix of assets with shape (N x N).

    Returns:
            The minus of expected return of the portfolio w plus the variance measurement RC(w).
                    Be aware that min -f(x) <=> max f(x).
    """
    return obj_Exp(w, mu) + obj_RC(w, p_cov)


def obj_Sharpe_Ratio(w, mu, p_cov):
    """Objective function for maximize the Sharpe Ratio given the portfolio w.

    Args:
            w : The portfolio of N assets with shape (N, ).
            mu : The expected return of each asset with shape (N, ).
            p_cov : The covariance matrix of assets with shape (N x N).

    Returns:
            The minus of Sharpe Ratio. Be aware that min -f(x) <=> max f(x).
    """

    return obj_Exp(w, mu) / obj_RC(w, p_cov)**0.5


def sum_weight_upper_bound(w, U):
    return U-np.sum(w)


def cons_sum_weight_upper_bound(U):
    """Return the constraint 1.T @ w <= U
    """
    return {'type': 'ineq', 'fun': sum_weight_upper_bound, 'args': [U]}


def sum_weight_lower_bound(w, L):
    return np.sum(w)-L


def cons_sum_weight_lower_bound(L):
    """Return the constraint 1.T @ w >= L
    """
    return {'type': 'ineq', 'fun': sum_weight_lower_bound, 'args': [L]}


def weight(w):
    return w


def cons_non_negative_weight():
    """Return the constraint w >= 0
    """
    return {'type': 'ineq', 'fun': weight}


def RC_upper_bound(w, p_cov, U):
    -w*(p_cov @ w)/(w.T @ p_cov @ w)+U


def cons_large_risk_constraint(p_cov, U):
    """Return the set of constraints: RCi <= Ui for all i=1, ..., N.
        I.e., the constraints about the marignal risk w.r.t. each stock (upper bound).
    """
    return {'type': 'ineq',
            'fun': RC_upper_bound, 'args': [p_cov, U]}


def RC_lower_bound(w, p_cov, L):
    w*(p_cov @ w)/(w.T @ p_cov @ w)-L


def cons_small_risk_constraint(p_cov, L):
    """Return the set of constraints: RCi >= Li for all i=1, ..., N.
        I.e., the constraints about the marignal risk w.r.t. each stock (lower bound).
    """
    return {'type': 'ineq',
            'fun': RC_lower_bound, 'args': [p_cov, L]}


# calculate risk budgeting portfolio weight given risk budget
def rb_p_weights(p_mean, p_cov, objective=obj_Exp_minus_RC, constraints=[cons_non_negative_weight()]):
    """_summary_

    Args:
            asset_rets (_type_): _description_
            objective (function): A valid objective function
            constraint (str): In the format of "#+#+#". E.g., "1+3" specifies the first and third type of constraints.

    Returns:
            _type_: _description_
    """
    # number of ARP series
    num_arp = p_mean.shape[0]
    w0 = np.ones(num_arp) / num_arp

    # Portfolio optimisation
    return minimize(objective, w0, args=(p_mean, p_cov), constraints=constraints)

# contruct portfolio


def portfolio_construction(momentum_period, rank, R_excess_df, momentum_atLeast=.001, num_atLeast=0, test_start_time=pd.Timestamp("2017"), objective=obj_Exp_minus_RC, constraints=[cons_non_negative_weight()]):
    """This function construct monthly updated portfolio. At each month, stocks with good momentum
        will be selected from the stock pool provided by R_df. Number of stocks to be selected is based on the pool size, rank parameter
        and the performance of these stocks. For those with negative momentum of excess return during the momentum period, the
        portfolio will give them 0 weight.

        Args:
            momentum_period (int): number of month to consider the momentum.
            rank (int): number of the best stocks to consider based on the momentum.
            R_excess_df (np.ndarray): The stock pool. R_df contains the excess return rates of all stocks.
            momentum_atLeast (float): Only invest the assets with momentum higher than this value.
            num_atLeast (int): The minimum number of assets at each month's investment.
            test_start_time (datetime, optional): The starting time of the test period in R_excess_df. Defaults to pd.Timestamp("2017").
            objective (function, optional): The objective function to minimize. Defaults to obj_Exp_minus_RC.
            constraints (list, optional): The list of constraints added to the optimization problem. Defaults to [cons_non_negative_weight()].

        Returns:
            R_excess_hat (np.ndarray): The estimated monthly excess return given by the monthly updated portfolio.
            w_hat (np.ndarray): The monthly updated portfolio we constructed.
    """
    # portfolio dates
    test_month = R_excess_df.index[R_excess_df.index >= test_start_time]

    # initialise portfolio return matrix
    R_excess_hat = np.zeros((test_month.shape[0], R_excess_df.shape[1]))
    w_hat = np.zeros((test_month.shape[0], R_excess_df.shape[1]))
    opt_results = np.empty(test_month.shape[0], dtype=object)

    for i, crt_month in enumerate(test_month):
        R_np = R_excess_df[R_excess_df.index <= crt_month].values

        momentum = R_np[-(momentum_period+1):-1, :].mean(axis=0)
        ranking_idx = np.argsort(momentum)[::-1]
        momentum = momentum[ranking_idx]

        numOfInterest = min((np.argmin(momentum > momentum_atLeast), rank, R_np.shape[1]))
        numOfInterest = max(num_atLeast, numOfInterest)
        if numOfInterest > num_atLeast:
            ranking_idx = ranking_idx[:numOfInterest]
            R_np = R_np[:, ranking_idx]
            crt_return = R_np[-1]

            R_train = R_np[:-1, :]

            # Todo: use different estimators for mu and C
            mu = R_train.mean(axis=0)
            C_hat = np.cov(R_train.T, ddof=0)
            if numOfInterest == 1:
                mu = np.array([mu])
                C_hat = np.array([[C_hat]])

            opt_results[i] = rb_p_weights(mu, C_hat, objective, constraints)
            if not opt_results[i].success:
                print("Warning! Fail to solve the optimization problem!")
            w_hat[i, ranking_idx] = opt_results[i].x
            R_excess_hat[i, ranking_idx] = w_hat[i, ranking_idx] * crt_return
        else:
            print("Forced least number of assets for investment at:", crt_month)
    return R_excess_hat, w_hat



def portfolio2(Bayes_df, R_excess_df, momentum_period=2, rank=100, momentum_atLeast=.001, num_atLeast=0, test_start_time=pd.Timestamp("2017"), objective=obj_Exp_minus_RC, constraints=[cons_non_negative_weight()]):
    """This function construct monthly updated portfolio using another approach.

        Args:
            momentum_period (int): number of month to consider the momentum.
            rank (int): number of the best stocks to consider based on the momentum.
            R_excess_df (np.ndarray): The stock pool. R_df contains the excess return rates of all stocks.
            momentum_atLeast (float): Only invest the assets with momentum higher than this value.
            num_atLeast (int): The minimum number of assets at each month's investment.
            test_start_time (datetime, optional): The starting time of the test period in R_excess_df. Defaults to pd.Timestamp("2017").
            objective (function, optional): The objective function to minimize. Defaults to obj_Exp_minus_RC.
            constraints (list, optional): The list of constraints added to the optimization problem. Defaults to [cons_non_negative_weight()].

        Returns:
            R_excess_hat (np.ndarray): The estimated monthly excess return given by the monthly updated portfolio.
            w_hat (np.ndarray): The monthly updated portfolio we constructed.
    """
    # portfolio dates
    test_month = R_excess_df.index[R_excess_df.index >= test_start_time]
    
    # initialise portfolio return matrix
    R_excess_hat = np.zeros((test_month.shape[0], R_excess_df.shape[1]))
    w_hat = np.zeros((test_month.shape[0], R_excess_df.shape[1]))
    opt_results = np.empty(test_month.shape[0], dtype=object)

    for i, crt_month in enumerate(test_month):
        R_np = R_excess_df[R_excess_df.index <= crt_month].values
        # momentum = R_np[-(momentum_period+1):-1, :].mean(axis=0)
        # ranking_idx = np.argsort(momentum)[::-1]
        numOfInterest = R_excess_df.shape[1]
        ranking_idx = np.arange(rank)
        # momentum = momentum[ranking_idx]

        # numOfInterest = min((np.argmin(momentum > momentum_atLeast), rank, R_np.shape[1]))
        # numOfInterest = max(num_atLeast, numOfInterest)
        if numOfInterest > num_atLeast:
            ranking_idx = ranking_idx[:numOfInterest]
            R_np = R_np[:, ranking_idx]
            crt_return = R_np[-1]

            R_train = R_np[:-1, :]

            # Here, we use Bayesian estimate for mu
            mu = Bayes_df.loc[crt_month, :]
            mu = mu.iloc[ranking_idx]
            # print('current month', crt_month, 'The Bayesian prediction of this month return')
            # print(mu)
            C_hat = np.cov(R_train.T, ddof=0)
            if numOfInterest == 1:
                mu = np.array([mu])
                C_hat = np.array([[C_hat]])

            opt_results[i] = rb_p_weights(mu, C_hat, objective, constraints)
            if not opt_results[i].success:
                print("Warning! Fail to solve the optimization problem!")
            w_hat[i, ranking_idx] = opt_results[i].x
            R_excess_hat[i, ranking_idx] = w_hat[i, ranking_idx] * crt_return
        else:
            print("Forced least number of assets for investment at:", crt_month)
    return R_excess_hat, w_hat
