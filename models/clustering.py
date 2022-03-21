import sys

sys.path.extend([".", ".."])
from CONSTANTS import *
import numpy as np
from numpy import linalg as LA
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist, cdist
from hdbscan import HDBSCAN as dbscan
from utils.common import metrics


def param_selection(trained_model, inputs, labels):
    '''
    Parameter selection for log clustering based approaches.
    :param trained_model: trained LogCluster model
    :param inputs: test inputs
    :param labels: test labels
    :return: selected best threshold.
    '''
    best_threshold = 0.0
    best_f1 = 0.0
    temp_thre = 0.0
    # step = 0.000005
    step = 0.000005
    while temp_thre <= 0.5:
        _, _, f1 = trained_model.evaluate(inputs, labels, temp_thre)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = temp_thre
        temp_thre += step

    return best_threshold




class LogClustering(object):
    """
    The implementation of Log Clustering model for anomaly detection.

    Authors:
        LogPAI Team

    Reference:
        [1] Qingwei Lin, Hongyu Zhang, Jian-Guang Lou, Yu Zhang, Xuewei Chen. Log Clustering
            based Problem Identification for Online Service Systems. International Conference
            on Software Engineering (ICSE), 2016.

    """

    # Dispose Loggers.
    LogClusteringLogger = logging.getLogger('LogClustering')
    LogClusteringLogger.setLevel(logging.DEBUG)
    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(name)s - " + SESSION + " - %(levelname)s: %(message)s"))

    file_handler = logging.FileHandler(os.path.join(LOG_ROOT, 'LogClustering.log'))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(name)s - " + SESSION + " - %(levelname)s: %(message)s"))

    LogClusteringLogger.addHandler(console_handler)
    LogClusteringLogger.addHandler(file_handler)
    LogClusteringLogger.info(
        'Construct logger for LogClustering succeeded, current working directory: %s, logs will be written in %s' %
        (os.getcwd(), LOG_ROOT))

    def __init__(self, max_dist=0.3, anomaly_threshold=0.3, mode='online', num_bootstrap_samples=1000):
        """
        Attributes
        ----------
            max_dist: float, the threshold to stop the clustering process
            anomaly_threshold: float, the threshold for anomaly detection
            mode: str, 'offline' or 'online' mode for clustering
            num_bootstrap_samples: int, online clustering starts with a bootstraping process, which
                determines the initial cluster representatives offline using a subset of samples 
            representatives: ndarray, the representative samples of clusters, of shape 
                num_clusters-by-num_events
            cluster_size_dict: dict, the size of each cluster, used to update representatives online 
        """

        self.logger = LogClustering.LogClusteringLogger
        self.max_dist = max_dist
        self.anomaly_threshold = anomaly_threshold
        self.mode = mode
        self.num_bootstrap_samples = num_bootstrap_samples
        self.representatives = list()
        self.cluster_size_dict = dict()
        self.normal_clusters = set()
        self.logger.info(
            'Model Parameters: max_dist = %.4f, anomaly_threshold = %.4f' % (self.max_dist, self.anomaly_threshold))

    def fit(self, X, labeled_idx=None):
        self.logger.info('====== Model summary ======')
        if self.mode == 'offline':
            # The offline mode can process about 10K samples only due to huge memory consumption.
            self._offline_clustering(X)
        elif self.mode == 'online':
            # Bootstrapping phase
            if self.num_bootstrap_samples > 0:
                X_bootstrap = X[0:self.num_bootstrap_samples, :]
                self._offline_clustering(X_bootstrap)
            # Online learning phase
            if X.shape[0] > self.num_bootstrap_samples:
                self._online_clustering(X)

    def predict(self, X, threshold=None):
        y_pred = np.zeros(X.shape[0])
        for i in range(X.shape[0]):
            min_dist, min_index = self._get_min_cluster_dist(X[i, :])
            if not threshold:
                threshold = self.anomaly_threshold
            if min_dist > threshold:
                y_pred[i] = 1
            elif len(self.normal_clusters) != 0:
                # When there are given labeled instances during fitting phase.
                if min_index not in self.normal_clusters:
                    y_pred[i] = 1
        return y_pred

    def evaluate(self, X, y_true, threshold):
        self.logger.info('====== Evaluation summary ======')
        self.logger.info('Threshold: %.7f' % threshold)
        y_pred = self.predict(X, threshold)
        precision, recall, f1 = metrics(y_pred, y_true)
        self.logger.info('Precision: {:.4f}, recall: {:.4f}, F1-measure: {:.4f}\n' \
                         .format(precision, recall, f1))
        return precision, recall, f1


    def _offline_clustering(self, X):
        self.logger.info('Starting offline clustering...')
        p_dist = pdist(X, metric=self._distance_metric)
        # 层次聚类每次合并两个族
        # https://blog.csdn.net/weixin_42887138/article/details/117708688

        Z = linkage(p_dist, 'complete')
        # 一个列表，对应了每个x属于哪一个族
        cluster_index = fcluster(Z, self.max_dist, criterion='distance')
        self._extract_representatives(X, cluster_index)
        self.logger.info('Processed {} instances.'.format(X.shape[0]))
        self.logger.info('Found {} clusters offline.\n'.format(len(self.representatives)))
        # print('The representive vectors are:')
        # pprint.pprint(self.representatives.tolist())

    def _extract_representatives(self, X, cluster_index):
        num_clusters = len(set(cluster_index))
        for clu in range(num_clusters):
            # 获取某个cluster的所有实例的索引
            clu_idx = np.argwhere(cluster_index == clu + 1)[:, 0]
            self.cluster_size_dict[clu] = clu_idx.shape[0]
            # 中心点作为该族的代表
            repre_center = np.average(X[clu_idx, :], axis=0)
            self.representatives.append(repre_center)

    def _online_clustering(self, X):
        self.logger.info("Starting online clustering...")
        for i in range(self.num_bootstrap_samples, X.shape[0]):
            if (i + 1) % 2000 == 0:
                self.logger.info('Processed {} instances.'.format(i + 1))
            instance_vec = X[i, :]
            if len(self.representatives) > 0:
                min_dist, clu_id = self._get_min_cluster_dist(instance_vec)
                if min_dist <= self.max_dist:
                    self.cluster_size_dict[clu_id] += 1
                    self.representatives[clu_id] = self.representatives[clu_id] \
                                                   + (instance_vec - self.representatives[clu_id]) \
                                                   / self.cluster_size_dict[clu_id]
                    continue
            self.cluster_size_dict[len(self.representatives)] = 1
            self.representatives.append(instance_vec)
        self.logger.info('Processed {} instances.'.format(X.shape[0]))
        self.logger.info('Found {} clusters online.\n'.format(len(self.representatives)))
        # print('The representive vectors are:')
        # pprint.pprint(self.representatives.tolist())

    def _distance_metric(self, x1, x2):
        norm = LA.norm(x1) * LA.norm(x2)
        distance = 1 - np.dot(x1, x2) / (norm + 1e-8)
        if distance < 1e-8:
            distance = 0
        if LA.norm(x1) < 1e-8 and LA.norm(x2) < 1e-8:
            distance = 0
        return distance

    def _get_min_cluster_dist(self, instance_vec):
        min_index = -1
        min_dist = float('inf')
        for i in range(len(self.representatives)):
            cluster_rep = self.representatives[i]
            dist = self._distance_metric(instance_vec, cluster_rep)
            if dist < 1e-8:
                min_dist = 0
                min_index = i
                break
            elif dist < min_dist:
                min_dist = dist
                min_index = i
        return min_dist, min_index

    def _construct_knowledge_base(self, labeled_normal_vec):
        for vec in labeled_normal_vec:
            _, normal_idx = self._get_min_cluster_dist(vec)
            self.normal_clusters.add(normal_idx)
        self.logger.info("%d clusters are normal." % len(self.normal_clusters))


class Solitary_HDBSCAN():
    HDBSCANLogger = logging.getLogger('Solitary_HDBSCAN')
    HDBSCANLogger.setLevel(logging.DEBUG)
    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(name)s - " + SESSION + " - %(levelname)s: %(message)s"))

    file_handler = logging.FileHandler(os.path.join(LOG_ROOT, 'Solitary_HDBSCAN.log'))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(name)s - " + SESSION + " - %(levelname)s: %(message)s"))

    HDBSCANLogger.addHandler(console_handler)
    HDBSCANLogger.addHandler(file_handler)
    HDBSCANLogger.info(
        'Construct logger for Solitary_HDBSCAN succeeded, current working directory: %s, logs will be written in %s' %
        (os.getcwd(), LOG_ROOT))
    def __init__(self, min_cluster_size, min_samples, mode='normal-only'):
        LOG_ROOT = GET_LOGS_ROOT()
        # Dispose Loggers.


        self.logger = Solitary_HDBSCAN.HDBSCANLogger
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples
        self.model = dbscan(algorithm='best',
                            min_cluster_size=self.min_cluster_size,
                            min_samples=self.min_samples if self.min_samples != -1 else None,
                            core_dist_n_jobs=10,
                            metric='euclidean')
        self.clusters = None
        self.cluster_central = []
        self.outliers = []
        self.labels = []
        self.normal_cores = set()
        self.anomalous_cores = set()
        self.mode = mode

    def fit_predict(self, inputs):
        self.logger.info('Start training model')
        start_time = time.time()
        self.labels = self.model.fit_predict(inputs).tolist()
        self.clusters = set(self.labels)
        self.outliers = self.model.outlier_scores_.tolist()
        self.logger.info('Get Total %d clusters in %.2fs' % (len(self.clusters), (time.time() - start_time)))
        return self.labels

    def fit(self, inputs):
        self.model.fit(inputs)
        pass

    def evaluate(self, inputs, ground_truth, normal_ids, label2id):
        all_predicted = [label2id[x] for x in self.predict(inputs, normal_ids)]
        assert len(all_predicted) == len(ground_truth)
        ground_truth_without_labeled_normal = []
        predicted_label_without_labeled_normal = []
        id = 0
        for label, gt in zip(all_predicted, ground_truth):
            if id not in normal_ids:
                ground_truth_without_labeled_normal.append(gt)
                predicted_label_without_labeled_normal.append(label)
            id += 1
        precision, recall, f = metrics(predicted_label_without_labeled_normal, ground_truth_without_labeled_normal)
        self.logger.info('Precision %.4f recall %.4f f-score %.4f ' % (precision, recall, f))
        return precision, recall, f
        pass

    def min_dist(self, source, target):
        min_dist = float("inf")
        for line in target:
            d = np.linalg.norm(source[0] - line)
            if d < min_dist:
                min_dist = d
                if min_dist == 0:
                    break
        return min_dist

    def predict(self, inputs, normal_ids):
        '''
        normal_ids are involved in inputs.
        :param inputs: all input reprs
        :param normal_ids: labeled normal indexes.
        :return: predicted label for each line of inputs, labeled normal ones included.
        '''
        predicted = []
        assert len(inputs) == len(self.labels)
        inputs = np.asarray(inputs, dtype=np.float)
        self.logger.info('Summarizing labeled normals and their reprs.')
        normal_matrix = []
        for id in normal_ids:
            normal_matrix.append(inputs[id, :])
            if self.labels[id] != -1:
                self.normal_cores.add(self.labels[id])
        self.logger.info('Normal clusters are: ' + str(self.normal_cores))
        normal_matrix = np.asarray(normal_matrix, dtype=np.float)
        self.logger.info('Shape of normal matrix: %d x %d' % (normal_matrix.shape[0], normal_matrix.shape[1]))

        by_normal_core_normal = 0
        by_normal_core_anomalous = 0
        by_dist_normal = 0
        by_dist_anomalous = 0

        for id, predict_cluster in enumerate(self.labels):
            if id in normal_ids:
                # Add labeled normals as predicted normals to formalize the output format for other modules.
                predicted.append('Normal')
                continue
            if predict_cluster in self.normal_cores:
                by_normal_core_normal += 1
                predicted.append('Normal')
            elif predict_cluster == -1:
                cur_repr = inputs[id]
                dists = cdist([cur_repr], normal_matrix)
                if dists.min() == 0:
                    by_dist_normal += 1
                    predicted.append('Normal')
                else:
                    by_dist_anomalous += 1
                    predicted.append('Anomalous')

                pass
            else:
                by_normal_core_anomalous += 1
                predicted.append('Anomalous')
        self.logger.info(
            'Found %d normal, %d anomalous by normal clusters' % (by_normal_core_normal, by_normal_core_anomalous))
        self.logger.info('Found %d normal, %d anomalous by minimum distances' % (by_dist_normal, by_dist_anomalous))
        return predicted

