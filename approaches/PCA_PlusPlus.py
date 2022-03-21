import sys

sys.path.extend([".", ".."])
from CONSTANTS import *
from preprocessing.preprocess import Preprocessor
from representations.templates.statistics import Simple_template_TF_IDF
from preprocessing.datacutter.SimpleCutting import cut_by_613
from representations.sequences.statistics import Sequential_Add
from models.pca import PCA_PlusPlus

if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--dataset', default='HDFS', type=str, help='Target dataset. Default HDFS')
    argparser.add_argument('--parser', default='IBM', type=str,
                           help='Select parser, please see parser list for detail. Default Official.')
    argparser.add_argument('--n_components', type=int, default=20,
                           help="Number of component after PCA, dynamic ratio if less than one.")
    argparser.add_argument('--threshold', type=float, default=None,
                           help="Anomaly Threshold parameter in PCA.")
    argparser.add_argument('--c_alpha', type=float, default=3.2905,
                           help="Anomaly Threshold parameter in PCA.")

    args, extra_args = argparser.parse_known_args()
    dataset = args.dataset
    parser = args.parser
    n_components = args.n_components if args.n_components else 0.95
    anomaly_threshold = args.threshold
    c_alpha = args.c_alpha

    template_encoder = Simple_template_TF_IDF()

    preprocessor = Preprocessor()
    train, _, test = preprocessor.process(dataset=dataset, parsing=parser, template_encoding=template_encoder.present,
                                          cut_func=cut_by_613)

    test_labels = [int(preprocessor.label2id[inst.label]) for inst in test]

    sequential_encoder = Sequential_Add(preprocessor.embedding)
    train_inputs = sequential_encoder.transform(train)
    test_inputs = sequential_encoder.transform(test)

    model = PCA_PlusPlus(n_components=n_components)
    model.fit(train_inputs)
    model.evaluate(test_inputs, test_labels, fixed_threshold=anomaly_threshold)

    model.logger.info('All done.')
