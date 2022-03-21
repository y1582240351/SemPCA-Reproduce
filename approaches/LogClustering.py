import sys

sys.path.extend([".", ".."])
from CONSTANTS import *
import argparse
from models.clustering import LogClustering
from preprocessing.preprocess import Preprocessor
from preprocessing.datacutter.SimpleCutting import cut_by_613
from representations.sequences.statistics import FeatureExtractor


def generate_inputs_and_labels(insts, label2idx):
    inputs = []
    labels = np.zeros(len(insts))
    for idx, inst in enumerate(insts):
        inputs.append([int(x) for x in inst.sequence])
        label = int(label2idx[inst.label])
        labels[idx] = label
    return inputs, labels


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--dataset', default='Spirit', type=str, help='Target dataset. Default HDFS')
    argparser.add_argument('--parser', default='IBM', type=str,
                           help='Select parser, please see parser list for detail. Default Official.')
    argparser.add_argument('--max_dist', type=float, default=0.3, help="Max Distance parameter in LogClustering.")
    argparser.add_argument('--anomaly_threshold', type=float, default=None,
                           help="Anomaly Threshold parameter in LogClustering.")
    argparser.add_argument('--save_results', type=bool, default=False,
                           help="Whether to save results.")
    args, extra_args = argparser.parse_known_args()

    dataset = args.dataset
    parser = args.parser
    max_dist = args.max_dist
    anomaly_threshold = args.anomaly_threshold
    save_results = args.save_results

    # Training, Validating and Testing instances.
    processor = Preprocessor()
    train, _, test = processor.process(dataset=dataset, parsing=parser, template_encoding=None,
                                       cut_func=cut_by_613)

    #  sample 50% of normal log sequences for training.
    train_normals = list(filter(lambda x: x.label == 'Normal', train))
    train_normals = random.sample(train_normals, int(0.5 * len(train_normals)))

    # Train feature representation using training set and update instances' representation.
    train_all_inputs, _ = generate_inputs_and_labels(train, processor.label2id)
    train_normals, _ = generate_inputs_and_labels(train_normals, processor.label2id)
    test_inputs, test_labels = generate_inputs_and_labels(test, processor.label2id)

    feature_representor = FeatureExtractor()
    train_all_inputs = feature_representor.fit_transform(np.asarray(train_all_inputs, dtype=object),
                                                         term_weighting='tf-idf')
    train_inputs = feature_representor.transform(np.asarray(train_normals, dtype=object))
    test_inputs = feature_representor.transform(np.asarray(test_inputs, dtype=object))

    model = LogClustering(max_dist=max_dist, anomaly_threshold=anomaly_threshold, mode='online')
    model.fit(X=train_inputs)
    predict_start = time.time()
    test_predicted = model.evaluate(test_inputs, test_labels, threshold=anomaly_threshold)
    predict_end = time.time()
