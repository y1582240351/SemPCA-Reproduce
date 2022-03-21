"""
The implementation of PCA model for anomaly detection.

Authors:
    LogPAI Team

Reference:
    [1] Wei Xu, Ling Huang, Armando Fox, David Patterson, Michael I. Jordan.
        Large-Scale System Problems Detection by Mining Console Logs. ACM
        Symposium on Operating Systems Principles (SOSP), 2009.

"""
import sys

sys.path.extend([".", ".."])
from CONSTANTS import *
import numpy as np
from utils.common import metrics


class PCA_PlusPlus(object):
    # Dispose Loggers.
    _logger = logging.getLogger('PCA_PlusPlus')
    _logger.setLevel(logging.DEBUG)
    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(name)s - " + SESSION + " - %(levelname)s: %(message)s"))

    file_handler = logging.FileHandler(os.path.join(LOG_ROOT, 'PCA_PlusPlus.log'))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(name)s - " + SESSION + " - %(levelname)s: %(message)s"))

    _logger.addHandler(console_handler)
    _logger.addHandler(file_handler)
    _logger.info(
        'Construct logger for PCA_PlusPlus succeeded, current working directory: %s, logs will be written in %s' %
        (os.getcwd(), LOG_ROOT))

    def __init__(self, n_components=0.95, threshold=None, c_alpha=3.2905):
        self.proj_C = None
        self.components = None
        self.n_components = n_components
        self.threshold = threshold
        self.c_alpha = c_alpha
        self.fixed_threshold = threshold

    @property
    def logger(self):
        return PCA._logger

    def fit(self, X):
        self.logger.info('====== Model summary ======')
        num_instances, num_events = X.shape
        X_cov = np.dot(X.T, X) / float(num_instances)
        U, sigma, V = np.linalg.svd(X_cov)
        n_components = self.n_components
        if n_components < 1:
            total_variance = np.sum(sigma)
            variance = 0
            for i in range(num_events):
                variance += sigma[i]
                if variance / total_variance >= n_components:
                    break
            n_components = i + 1

        P = U[:, :n_components]
        I = np.identity(num_events, int)
        self.components = P
        self.proj_C = I - np.dot(P, P.T)
        self.logger.info('n_components: {}'.format(n_components))
        self.logger.info('Project matrix shape: {}-by-{}'.format(self.proj_C.shape[0], self.proj_C.shape[1]))

        if not self.threshold:
            phi = np.zeros(3)
            for i in range(3):
                for j in range(n_components, num_events):
                    phi[i] += np.power(sigma[j], i + 1)
            h0 = 1.0 - 2 * phi[0] * phi[2] / (3.0 * phi[1] * phi[1])
            self.threshold = phi[0] * np.power(self.c_alpha * np.sqrt(2 * phi[1] * h0 * h0) / phi[0]
                                               + 1.0 + phi[1] * h0 * (h0 - 1) / (phi[0] * phi[0]),
                                               1.0 / h0)
        self.logger.info('SPE threshold: {}\n'.format(self.threshold))

    def predict(self, X, fixed_threshold=None):
        assert self.proj_C is not None, 'PCA model needs to be trained before prediction.'
        y_pred = np.zeros(X.shape[0])
        for i in range(X.shape[0]):
            y_a = np.dot(self.proj_C, X[i, :])
            SPE = np.dot(y_a, y_a)
            if fixed_threshold is not None and fixed_threshold != -1.0:
                self.fixed_threshold = fixed_threshold
            else:
                self.fixed_threshold = self.threshold
            if SPE > self.fixed_threshold:
                y_pred[i] = 1
        return y_pred

    def evaluate(self, X, y_true, fixed_threshold=None):
        if fixed_threshold is not None:
            self.logger.info('Threshold: %.8f' % fixed_threshold)
        else:
            self.logger.info('No given threshold, will be using default PCA threshold: %.8f.' % self.threshold)
            fixed_threshold = self.threshold
        y_pred = self.predict(X, fixed_threshold)
        TP, TN, FP, FN = 0, 0, 0, 0
        for pred, true in zip(y_pred, y_true):
            if pred == true:
                if pred == 1:
                    TP += 1
                else:
                    TN += 1
            else:
                if pred == 1:
                    FP += 1
                else:
                    FN += 1
        if TP == 0:
            precision, recall, f1 = 0, 0, 0
        else:
            precision, recall, f1 = metrics(y_pred, y_true)
        self.logger.info(
            'Precision: {:.0f}/{:.0f} = {:.4f}, recall: {:.0f}/{:.0f} = {:.4f}, F1-measure: {:.4f}'.format(TP,
                                                                                                           (TP + FP),
                                                                                                           precision,
                                                                                                           TP,
                                                                                                           (TP + FN),
                                                                                                           recall, f1))

        return precision, recall, f1


class PCA(object):
    # Dispose Loggers.
    _logger = logging.getLogger('PCA')
    _logger.setLevel(logging.DEBUG)
    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(name)s - " + SESSION + " - %(levelname)s: %(message)s"))

    file_handler = logging.FileHandler(os.path.join(LOG_ROOT, 'PCA.log'))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(name)s - " + SESSION + " - %(levelname)s: %(message)s"))

    _logger.addHandler(console_handler)
    _logger.addHandler(file_handler)
    _logger.info(
        'Construct logger for PCA succeeded, current working directory: %s, logs will be written in %s' %
        (os.getcwd(), LOG_ROOT))

    def __init__(self, n_components=0.95, threshold=None, c_alpha=3.2905):
        """ The PCA model for anomaly detection

        Attributes
        ----------
            proj_C: The projection matrix for projecting feature vector to abnormal space
            n_components: float/int, number of principal compnents or the variance ratio they cover
            threshold: float, the anomaly detection threshold. When setting to None, the threshold 
                is automatically caculated using Q-statistics
            c_alpha: float, the c_alpha parameter for caculating anomaly detection threshold using 
                Q-statistics. The following is lookup table for c_alpha:
                c_alpha = 1.7507; # alpha = 0.08
                c_alpha = 1.9600; # alpha = 0.05
                c_alpha = 2.5758; # alpha = 0.01
                c_alpha = 2.807; # alpha = 0.005
                c_alpha = 2.9677;  # alpha = 0.003
                c_alpha = 3.2905;  # alpha = 0.001
                c_alpha = 3.4808;  # alpha = 0.0005
                c_alpha = 3.8906;  # alpha = 0.0001
                c_alpha = 4.4172;  # alpha = 0.00001
        """

        self.proj_C = None
        self.components = None
        self.n_components = n_components
        self.threshold = threshold
        self.c_alpha = c_alpha
        self.fixed_threshold = threshold

    @property
    def logger(self):
        return PCA._logger

    def fit(self, X):
        """
        Auguments
        ---------
            X: ndarray, the event count matrix of shape num_instances-by-num_events
        """

        self.logger.info('====== Model summary ======')
        num_instances, num_events = X.shape
        X_cov = np.dot(X.T, X) / float(num_instances)
        U, sigma, V = np.linalg.svd(X_cov)
        n_components = self.n_components
        if n_components < 1:
            total_variance = np.sum(sigma)
            variance = 0
            for i in range(num_events):
                variance += sigma[i]
                if variance / total_variance >= n_components:
                    break
            n_components = i + 1

        P = U[:, :n_components]
        I = np.identity(num_events, int)
        self.components = P
        self.proj_C = I - np.dot(P, P.T)
        self.logger.info('n_components: {}'.format(n_components))
        self.logger.info('Project matrix shape: {}-by-{}'.format(self.proj_C.shape[0], self.proj_C.shape[1]))

        if not self.threshold:
            # Calculate threshold using Q-statistic. Information can be found at:
            # http://conferences.sigcomm.org/sigcomm/2004/papers/p405-lakhina111.pdf
            phi = np.zeros(3)
            for i in range(3):
                for j in range(n_components, num_events):
                    phi[i] += np.power(sigma[j], i + 1)
            h0 = 1.0 - 2 * phi[0] * phi[2] / (3.0 * phi[1] * phi[1])
            self.threshold = phi[0] * np.power(self.c_alpha * np.sqrt(2 * phi[1] * h0 * h0) / phi[0]
                                               + 1.0 + phi[1] * h0 * (h0 - 1) / (phi[0] * phi[0]),
                                               1.0 / h0)
        self.logger.info('SPE threshold: {}\n'.format(self.threshold))

    def predict(self, X, fixed_threshold=None):
        assert self.proj_C is not None, 'PCA model needs to be trained before prediction.'
        y_pred = np.zeros(X.shape[0])
        for i in range(X.shape[0]):
            y_a = np.dot(self.proj_C, X[i, :])
            SPE = np.dot(y_a, y_a)
            if fixed_threshold is not None and fixed_threshold != -1.0:
                self.fixed_threshold = fixed_threshold
            else:
                self.fixed_threshold = self.threshold
            if SPE > self.fixed_threshold:
                y_pred[i] = 1
        return y_pred

    def evaluate(self, X, y_true, fixed_threshold=None):
        if fixed_threshold is not None:
            self.logger.info('Threshold: %.8f' % fixed_threshold)
        else:
            self.logger.info('No given threshold, will be using default PCA threshold: %.8f.' % self.threshold)
            fixed_threshold = self.threshold
        y_pred = self.predict(X, fixed_threshold)
        TP, TN, FP, FN = 0, 0, 0, 0
        for pred, true in zip(y_pred, y_true):
            if pred == true:
                if pred == 1:
                    TP += 1
                else:
                    TN += 1
            else:
                if pred == 1:
                    FP += 1
                else:
                    FN += 1
        if TP == 0:
            precision, recall, f1 = 0, 0, 0
        else:
            precision, recall, f1 = metrics(y_pred, y_true)
        self.logger.info(
            'Precision: {:.0f}/{:.0f} = {:.4f}, recall: {:.0f}/{:.0f} = {:.4f}, F1-measure: {:.4f}'.format(TP,
                                                                                                           (TP + FP),
                                                                                                           precision,
                                                                                                           TP,
                                                                                                           (TP + FN),
                                                                                                           recall, f1))

        return precision, recall, f1
