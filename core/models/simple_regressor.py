import numpy as np
from sklearn.linear_model import LinearRegression
from numpy.linalg import LinAlgError

class LinearRegressor:
    def __init__(self, X, y, label=""):
        y = y - y[0]
        self.X = np.array([[x] for x in X])
        #self.X = np.array(X).reshape((len(X[0]), 1))
        #self.y = y # np.array([[_y] for _y in y])
        self.y = np.array([[_y] for _y in y])

    def train(self):
        self.regression_line = LinearRegression(fit_intercept=False, positive=True).fit(self.X, self.y)

    # Expected 2d numpy array
    def predict(self, x):
        x = np.array([[_x] for _x in [x]])
        return self.regression_line.predict(x)


class NonLinearRegressor:
    def __init__(self, X, y, label=""):
        y = np.reshape(y, newshape=(y.shape[-1]))
        #y = y - y[0] # Adjusting so that intercept is always 0
        #self.X = np.array([[x] for x in X])
        #self.y = np.array([[_y] for _y in y])
        self.X = X
        self.y = y
        self.label = label

    def train(self, train_percentage=0.6, label=""):
        cut = train_percentage
        dataset = self.X

        indexes = np.arange(dataset.shape[0])
        np.random.shuffle(indexes)
        train_index = indexes[: int(cut * dataset.shape[0])]
        val_index = indexes[int(cut * dataset.shape[0]):]
        train_x = dataset[train_index]
        val_x = dataset[val_index]

        train_y = self.y[train_index]
        val_y = self.y[val_index]


        best_error = float('inf')
        best_degree = -1
        best_model = None
        for degree in range(1, 10):
            #coef_list = self.__simple_polyfit(train_x, train_y, degree)
            #poly_model = np.poly1d(coef_list)
            poly_model = self.__polyfit_fixed_on_all(train_x, train_y, degree)
            error = self.__evaluate(poly_model, val_x, val_y)
            if error < best_error:
                best_error = error
                best_degree = degree
                best_model = poly_model
            #self.__visualize_graph(train_x, train_y, val_x, val_y,
            #                       poly_model, degree, error)

        #self.__visualize_graph(train_x, train_y, val_x, val_y,
        #                       best_model, best_degree, best_error)
        self.poly_model = best_model

    def __simple_polyfit(self, train_x, train_y, degree):
        return np.polyfit(train_x, train_y, degree)

    def __polyfit_fixed_on_zero(self, train_x, train_y, degree):
        n, d, f = len(train_x), degree, 1
        x = train_x
        xf = np.array([0])
        y = train_y
        yf = np.array([0])
        params = self.__polyfit_with_fixed_points(d, x, y, xf, yf)
        poly = np.polynomial.Polynomial(params)
        return poly

    def __polyfit_fixed_on_all(self, train_x, train_y, degree):
        n, d, f = len(train_x), degree, len(train_x)
        x = train_x
        xf = self.X[:degree]
        y = train_y
        yf = self.y[:degree]
        params = self.__polyfit_with_fixed_points(d, x, y, xf, yf)
        poly = np.polynomial.Polynomial(params)
        return poly

    def __polyfit_with_fixed_points(self, n, x, y, xf, yf):
        mat = np.empty((n + 1 + len(xf),) * 2)
        vec = np.empty((n + 1 + len(xf),))
        x_n = x**np.arange(2 * n + 1)[:, None]
        yx_n = np.sum(x_n[:n + 1] * y, axis=1)
        x_n = np.sum(x_n, axis=1)
        idx = np.arange(n + 1) + np.arange(n + 1)[:, None]
        mat[:n + 1, :n + 1] = np.take(x_n, idx)
        xf_n = xf**np.arange(n + 1)[:, None]
        mat[:n + 1, n + 1:] = xf_n / 2
        mat[n + 1:, :n + 1] = xf_n.T
        mat[n + 1:, n + 1:] = 0
        vec[:n + 1] = yx_n
        vec[n + 1:] = yf
        params = np.linalg.solve(mat, vec)
        return params[:n + 1]

    def __visualize_graph(self, train_x, train_y, val_x, val_y,
                             poly_model, degree, error):
        from matplotlib import pyplot as plt
        x_lim = train_x.max()
        xx = np.linspace(0, x_lim, int(round(x_lim / 50, 0)))
        plt.plot(train_x, train_y, 'bo')
        plt.plot(val_x, val_y, 'go')
        plt.plot([0], [0], 'ro')
        plt.plot(xx, poly_model(xx), '-')

        # plt.text(0, 0, "Degree: " + str(degree), fontsize=15)
        textstr = "Degree: " + str(degree) + "    "
        textstr += "Val Error: " + str(round(error, 2)) + "    "
        textstr += "Coefficients: " + str(np.round(poly_model.coef, 2))
        plt.text(0.02, 0.92, textstr, fontsize=12, transform=plt.gcf().transFigure)
        textstr = self.label
        plt.text(0.02, 0.97, textstr, fontsize=10, transform=plt.gcf().transFigure)
        plt.show()



    def __evaluate(self, poly_model, val_x, val_y):
        pred_y = poly_model(val_x)
        rmse = np.sqrt(np.mean((pred_y-val_y)**2))
        return rmse

    # Expected 2d numpy array
    def predict(self, x):
        return self.poly_model(x)

class FrameRegressor:
    def __init__(self, X):
        self.X = X

        shape = X.shape
        seq_size = shape[0]
        self.regressor_matrix = []
        for i in range(shape[1]):
            regressor_row = []
            for j in range(shape[2]):
                l = LinearRegressor([range(seq_size)], X[:, i, j])
                regressor_row.append(l)
            self.regressor_matrix.append(regressor_row)

    def train(self):
        for row in self.regressor_matrix:
            for regressor in row:
                regressor.train()

    # Expected 3d numpy array
    def predict(self, size_ahead):
        shape = self.X.shape
        start = shape[0] + 1
        x = range(start, start + size_ahead)
        y = np.zeros(shape = (size_ahead, self.X.shape[1], self.X.shape[2]))

        for i in range(shape[1]):
            for j in range(shape[2]):
                regressor = self.regressor_matrix[i][j]
                line = regressor.predict(x)
                y[:, i, j] = line.reshape(line.shape[0])
        return y


def test_non_linear_regressor_on_parabola():
    nl = NonLinearRegressor(np.array([0, 1, 2, 3, 4, 5, 6]), np.array([0, 1, 4, 9, 16, 32, 64]))
    nl.train()
    nl.predict(np.array([2.5]))

def test_non_linear_regressor_on_model_data():
    parameters_file = "/home/anderson/Programacao/DJEnsemble/Stream-DJEnsemble/models/spatio-temporal/cfsr-all/"
    model_name = "CFSR-2014.nc-x0=(115, 129)-3x3-('2014-01-01 00:00:00', '2014-03-30 23:45:00')-summer-noise_level-50.parameters"
    #parameters_file += "CFSR-2014.nc-x0=(80, 9)-7x7-('2014-01-01 00:00:00', '2014-03-30 23:45:00')-summer-noise_level-50.parameters"
    parameters_file += model_name
    with open(parameters_file) as f:
        distances = f.readline().split("#")[1]  # [:-1]
        error = f.readline().split("#")[1]  # [:-1]
        distances = eval(distances.strip())
        error = eval(error.strip())

    nl = NonLinearRegressor(np.array(distances), np.array([error]), model_name)
    nl.train()
    print(nl.predict(np.array([2.5])))


if __name__ == '__main__':
    #test_non_linear_regressor_on_parabola()
    test_non_linear_regressor_on_model_data()


# Example
# X = np.array([[1], [3], [5], [7]])
# y = np.array([[10], [30], [50], [70]])
# r = LinearRegressionStrategy(X, y)
# r.train()
# print(r.predict(np.array([[20]])))

# reg = LinearRegression().fit(X, y)
# reg.score(X, y)
# print(reg.predict(np.array([[5]])))