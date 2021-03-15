# Copyright (c) 2016, the GPyOpt Authors
# Licensed under the BSD 3-clause license (see LICENSE.txt)

import numpy as np
from pylab import grid
import matplotlib.pyplot as plt
from pylab import savefig
import pylab


def plot_acquisition(bounds, input_dim, model, Xdata, Ydata, acquisition_function, suggested_sample, fixed_values,
                     filename=None, label_x=None, label_y=None, color_by_step=True):
    '''
    Plots of the model and the acquisition function in 1D and 2D examples.
    '''

    if fixed_values is None:
        fixed_values = [None] * input_dim

    free_dimensions = []
    for i, value in enumerate(fixed_values):
        if value is None:
            free_dimensions.append(i)

    assert len(free_dimensions) == 2 or len(free_dimensions) == 1, \
        "In case of dimension being more than 2 fixed_values must " \
        "be specified so as to make a number of free dimensions equal 2 or 1."

    if len(free_dimensions) == 1:
        index_x = free_dimensions[0]

        if not label_x:
            label_x = f'X{index_x + 1}'

        X_values = np.arange(bounds[index_x][0], bounds[index_x][1], 0.001).flatten()
        n = len(X_values)
        X = np.zeros((n, input_dim))
        for i, value in enumerate(fixed_values):
            if value is not None:
                X[:, i] = [value] * n
        X[:, index_x] = X_values
        X_values = X_values.reshape((n, 1))

        acqu = acquisition_function(X)
        acqu_normalized = (-acqu - min(-acqu)) / (max(-acqu - min(-acqu)))  # normalize acquisition
        m, v = model.predict(X)
        plt.ioff()
        plt.figure(figsize=(10, 5))
        plt.subplot(2, 1, 1)
        plt.plot(X_values, m, 'b-', label=u'Posterior mean', lw=2)
        plt.fill(np.concatenate([X_values, X_values[::-1]]),
                 np.concatenate([m - 1.9600 * np.sqrt(v),
                                 (m + 1.9600 * np.sqrt(v))[::-1]]),
                 alpha=.5, fc='b', ec='None', label='95% C. I.')
        plt.plot(X_values, m - 1.96 * np.sqrt(v), 'b-', alpha=0.5)
        plt.plot(X_values, m + 1.96 * np.sqrt(v), 'b-', alpha=0.5)
        plt.plot(Xdata[:, index_x], Ydata, 'r.', markersize=10, label=u'Observations')
        plt.axvline(x=suggested_sample[len(suggested_sample) - 1][index_x], color='r')
        plt.title('Model and observations')
        plt.ylabel('f(x)')
        plt.xlabel(label_x)
        plt.legend(loc='upper left')
        plt.xlim(*bounds[index_x])
        grid(True)
        plt.subplot(2, 1, 2)
        plt.axvline(x=suggested_sample[len(suggested_sample) - 1][index_x], color='r')
        plt.plot(X_values, acqu_normalized, 'r-', lw=2)
        plt.xlabel(label_x)
        plt.ylabel('Acquisition value')
        plt.title('Acquisition function')
        grid(True)
        plt.xlim(*bounds[index_x])

    if (len(free_dimensions) == 2):
        index_x, index_y = free_dimensions

        if not label_x:
            label_x = f'X{index_x + 1}'

        if not label_y:
            label_y = f'X{index_y + 1}'

        n = Xdata.shape[0]
        colors = np.linspace(0, 1, n)
        cmap = plt.cm.Reds
        norm = plt.Normalize(vmin=0, vmax=1)
        points_var_color = lambda X: plt.scatter(
            X[:, index_x], X[:, index_y], c=colors, label=u'Observations', cmap=cmap, norm=norm)
        points_one_color = lambda X: plt.plot(
            X[:, index_x], X[:, index_y], 'r.', markersize=10, label=u'Observations')
        X1 = np.linspace(bounds[index_x][0], bounds[index_x][1], 200)
        X2 = np.linspace(bounds[index_y][0], bounds[index_y][1], 200)
        x1, x2 = np.meshgrid(X1, X2)
        grid_point_num = 200 * 200

        X = np.zeros((grid_point_num, input_dim))

        for i, value in enumerate(fixed_values):
            if value is not None:
                X[:, i] = [value] * grid_point_num

        X[:, index_x] = x1.reshape(200 * 200)
        X[:, index_y] = x2.reshape(200 * 200)
        acqu = acquisition_function(X)
        acqu_normalized = (-acqu - min(-acqu)) / (max(-acqu - min(-acqu)))
        acqu_normalized = acqu_normalized.reshape((200, 200))
        m, v = model.predict(X)
        plt.figure(figsize=(15, 5))
        plt.subplot(1, 3, 1)
        plt.contourf(X1, X2, m.reshape(200, 200), 100)
        plt.colorbar()
        if color_by_step:
            points_var_color(Xdata)
        else:
            points_one_color(Xdata)
        plt.ylabel(label_y)
        plt.title('Posterior mean')
        plt.axis((bounds[index_x][0], bounds[index_x][1], bounds[index_y][0], bounds[index_y][1]))
        ##
        plt.subplot(1, 3, 2)
        plt.contourf(X1, X2, np.sqrt(v.reshape(200, 200)), 100)
        plt.colorbar()
        if color_by_step:
            points_var_color(Xdata)
        else:
            points_one_color(Xdata)
        plt.xlabel(label_x)
        plt.ylabel(label_y)
        plt.title('Posterior sd.')
        plt.axis((bounds[index_x][0], bounds[index_x][1], bounds[index_y][0], bounds[index_y][1]))
        ##
        plt.subplot(1, 3, 3)
        plt.contourf(X1, X2, acqu_normalized, 100)
        plt.colorbar()
        plt.plot(suggested_sample[:, index_x], suggested_sample[:, index_y], 'm.', markersize=10)
        plt.xlabel(label_x)
        plt.ylabel(label_y)
        plt.title('Acquisition function')
        plt.axis((bounds[index_x][0], bounds[index_x][1], bounds[index_y][0], bounds[index_y][1]))
    if filename != None:
        savefig(filename)
    else:
        plt.show()


def plot_convergence(Xdata, best_Y, filename=None):
    '''
    Plots to evaluate the convergence of standard Bayesian optimization algorithms
    '''
    n = Xdata.shape[0]
    aux = (Xdata[1:n, :] - Xdata[0:n - 1, :]) ** 2
    distances = np.sqrt(aux.sum(axis=1))

    ## Distances between consecutive x's
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(list(range(n - 1)), distances, '-ro')
    plt.xlabel('Iteration')
    plt.ylabel('d(x[n], x[n-1])')
    plt.title('Distance between consecutive x\'s')
    grid(True)

    # Estimated m(x) at the proposed sampling points
    plt.subplot(1, 2, 2)
    plt.plot(list(range(n)), best_Y, '-o')
    plt.title('Value of the best selected sample')
    plt.xlabel('Iteration')
    plt.ylabel('Best y')
    grid(True)

    if filename != None:
        savefig(filename)
    else:
        plt.show()
