import numpy as np
import scipy.optimize as opt
import scipy.signal as sig

def residuals(p, func, x, y, s=1):
    """Find the residuals from fitting data to a function.
    
    Keyword arguments:
    p -- initial values of the coefficients to be fit
    x -- the x data to be fit
    y -- the y data to be fit
    func -- the function to be fit
    s -- the standard deviation (default 1)
    """
    return (y - func(p,x)) / s

# Fitting function
def fitting(p, x, y, func, s=1):
    """
    Fit data to a function.
    
    Parameters
    ----------
    p : array_like
        Initial guess at the values of the coefficients to be fit
    x : array_like
        The x data to be fit.
    y : array_like
        The y data to be fit.
    func : callable
        The function to fit the data to.
    s : float, default 1
        The standard deviation.

    Returns
    -------
    p_fit : numpy.ndarray
        The array of the coefficients after fitting the function to the data.
    chi : float
        The chi squared value of this fit.
    unc_fit : numpy.ndarray
        The array of the uncertainties in the values of p_fit.
    """
    # Fit the data and find the uncertainties
    r = opt.least_squares(residuals, p, args=(func, x, y, s))
    p_fit = r.x
    hessian = np.dot(r.jac.T, r.jac) #estimate the hessian matrix
    K_fit = np.linalg.inv(hessian) #covariance matrix
    unc_fit = np.sqrt(np.diag(K_fit)) #stdevs
    
    # rescale
    beta = np.sqrt(np.sum(residuals(p_fit, func, x, y)**2) / (x.size - len(p_fit)))
    unc_fit = unc_fit * beta
    K_fit = K_fit * beta
    
    # find the chi2 score
    chi = np.sum(residuals(p_fit, func, x, y, beta)**2) / (x.size - len(p_fit))
    
    return p_fit, chi, unc_fit

def gauss(p, x):
    """
    Produce a Gaussian function.
    
    Parameters
    ----------
    p : array_like
        Coefficients of the Gaussian function.
    x : array_like
        The x range over which the Gaussian function is to be produced.

    Returns
    -------
    numpy.ndarray
        The Gaussian function.
    """
    return p[0] * np.exp(-(x - p[1])**2 / (2 * p[2]**2)) + p[3]

def lorentz(p, x):
    """
    Produce a Lorentzian function.
    
    Parameters
    ----------
    p : array_like
        Coefficients of the Lorentzian function.
    x : array_like
        The x range over which the Lorentzian function is to be produced.

    Returns
    -------
    numpy.ndarray
        The Lorentzian function.
    """
    return p[0] * p[2]**2 / ((x - p[1])**2 + p[2]**2) + p[3]

def sin(p, x):
    """
    Produce a sin function.
    
    Parameters
    ----------
    p : array_like
        Coefficients of the sin function.
    x : array_like
        The x range over which the sin function is to be produced.

    Returns
    -------
    numpy.ndarray
        The sin function.
    """
    return p[0]*np.sin(p[1]*x+p[2]) + p[3]

def cos_superpos(p, x):
    """
    Produce a function that is the sum of two cos functions.
    
    Parameters
    ----------
    p : array_like
        Coefficients of the function.
    x : array_like
        The x range over which the function is to be produced.

    Returns
    -------
    numpy.ndarray
        The function.
    """
    return p[0]*(np.cos(p[1]*x+p[2]) + np.cos(p[3]*x+p[4])) + p[5]

def cos_prod(p, x):
    """
    Produce a function that is the product of two cos functions.
    
    Parameters
    ----------
    p : array_like
        Coefficients of the function.
    x : array_like
        The x range over which the function is to be produced.

    Returns
    -------
    numpy.ndarray
        The function.
    """
    return p[0]*np.cos(p[1]*x+p[2]) * np.cos(p[3]*x+p[4]) + p[5]

def round_sig_fig_uncertainty(value, uncertainty):
    """
    Round to the first significant figure of the uncertainty.
    
    Parameters
    ----------
    value : float or array_like
        The value(s) to be rounded.
    uncertainty : float or array_like
        The uncertaint(y/ies) in this value, must be the same size as value.

    Returns
    -------
    value_out : numpy.ndarray or float
        The rounded array of values.
    uncertainty_out : numpy.ndarray or float
        The rounded array of uncertainties.

    See Also
    --------
    round_sig_fig : Round to a given number of significant figures.
    """
    # check if numpy array/list or float/int
    if isinstance(value, np.ndarray) or isinstance(value, list):
        value_out = np.array([])
        uncertainty_out = np.array([])
        for i in range(len(value)):
            # Check if some of the values are 0
            if uncertainty[i] == 0:
                value_out = np.append(value_out, value[i])
                uncertainty_out = np.append(uncertainty_out, uncertainty[i])
            # Check if the leading digit in the error is 1, and if so round to an extra significant figure
            elif np.floor(uncertainty[i] / (10**np.floor(np.log10(uncertainty[i])))) != 1.0:
                uncertainty_rnd = np.round(uncertainty[i], int(-(np.floor(np.log10(uncertainty[i])))))
                value_rnd = np.round(value[i], int(-(np.floor(np.log10(uncertainty_rnd)))))

                value_out = np.append(value_out, value_rnd)
                uncertainty_out = np.append(uncertainty_out, uncertainty_rnd)
            else:
                uncertainty_rnd = np.round(uncertainty[i], int(1 - (np.floor(np.log10(uncertainty[i])))))
                value_rnd = np.round(value[i], int(1 - (np.floor(np.log10(uncertainty_rnd)))))

                value_out = np.append(value_out, value_rnd)
                uncertainty_out = np.append(uncertainty_out, uncertainty_rnd)
        return value_out, uncertainty_out
   
    elif isinstance(value, float) or isinstance(value, int):
        if uncertainty == 0:
            return value, uncertainty
        elif np.floor(uncertainty / (10**np.floor(np.log10(uncertainty)))) != 1.0:
            uncertainty_out = np.round(uncertainty, int(-(np.floor(np.log10(uncertainty)))))
            value_out = np.round(value, int(-(np.floor(np.log10(uncertainty_out)))))

            return value_out, uncertainty_out
        else:
            uncertainty_out = np.round(uncertainty, int(1 - (np.floor(np.log10(uncertainty)))))
            value_out = np.round(value, int(1 - (np.floor(np.log10(uncertainty_out)))))

            return value_out, uncertainty_out
    else:
        return value, uncertainty