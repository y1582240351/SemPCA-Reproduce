import sys



sys.path.extend([".", ".."])
from CONSTANTS import *
from representations.sequences.statistics import Sequential_Add
from representations.templates.statistics import Simple_template_TF_IDF
from module.Optimizer import Optimizer
from preprocessing.datacutter.SimpleCutting import cut_by_613
from preprocessing.prob_labeling import Probabilistic_Labeling
import argparse
from sklearn.decomposition import FastICA
from preprocessing.preprocess import Preprocessor
from utils.common import get_precision_recall, data_iter, generate_tinsts_binary_label, batch_variable_inst
from models.lstm import PLELog
# Hyper parameters.
lstm_hiddens = 100
num_layer = 2
batch_size = 100
epochs = 5


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--dataset', default='HDFS', type=str, help='Target dataset. Default HDFS')
    argparser.add_argument('--mode', default='train', type=str, help='train or test')
    argparser.add_argument('--parser', default='IBM', type=str,
                           help='Select parser, please see parser list for detail. Default Official.')
    argparser.add_argument('--min_cluster_size', type=int, default=100,
                           help="min_cluster_size.")
    argparser.add_argument('--min_samples', type=int, default=100,
                           help="min_samples")
    argparser.add_argument('--reduce_dimension', type=int, default=50,
                           help="Reduce dimentsion for fastICA, to accelerate the HDBSCAN probabilistic label estimation.")
    argparser.add_argument('--threshold', type=float, default=0.5,
                           help="Anomaly threshold for PLELog.")
    args, extra_args = argparser.parse_known_args()

    dataset = args.dataset
    parser = args.parser
    mode = args.mode
    min_cluster_size = args.min_cluster_size
    min_samples = args.min_samples
    reduce_dimension = args.reduce_dimension
    threshold = args.threshold

    # Mark results saving directories.
    save_dir = os.path.join(PROJECT_ROOT, 'outputs')
    base = os.path.join(PROJECT_ROOT, 'datasets/' + dataset)
    output_model_dir = os.path.join(save_dir, 'models/PLELog/' + dataset + '_' + parser + '/model')
    output_res_dir = os.path.join(save_dir, 'results/PLELog/' + dataset + '_' + parser + '/detect_res')
    prob_label_res_file = os.path.join(save_dir,
                                       'results/PLELog/' + dataset + '_' + parser +
                                       '/prob_label_res/mcs-' + str(min_cluster_size) + '_ms-' + str(min_samples))
    rand_state = os.path.join(save_dir,
                              'results/PLELog/' + dataset + '_' + parser +
                              '/prob_label_res/random_state')

    save_dir = os.path.join(PROJECT_ROOT, 'outputs')

    template_encoder = Simple_template_TF_IDF()

    # Training, Validating and Testing instances.
    processor = Preprocessor()
    train, dev, test = processor.process(dataset=dataset, parsing=parser, cut_func=cut_by_613,
                                         template_encoding=template_encoder.present)

    # Log sequence representation.
    sequential_encoder = Sequential_Add(processor.embedding)
    train_reprs = sequential_encoder.present(train)
    for index, inst in enumerate(train):
        inst.repr = train_reprs[index]
    dev_reprs = sequential_encoder.present(dev)
    for index, inst in enumerate(dev):
        inst.repr = dev_reprs[index]
    test_reprs = sequential_encoder.present(test)
    for index, inst in enumerate(test):
        inst.repr = test_reprs[index]

    # Sample normal instances.
    train_normal = [x for x, inst in enumerate(train) if inst.label == 'Normal']
    normal_ids = train_normal[:int(0.5 * len(train_normal))]
    label_generator = Probabilistic_Labeling(min_samples=min_samples, min_clust_size=min_cluster_size,
                                             res_file=prob_label_res_file, rand_state_file=rand_state)

    # Load Embeddings
    embedding = []
    word_dims = 0
    for _, embed in processor.embedding.items():
        embedding.append(embed)
    word_dims = embedding[0].shape

    # Append padding (unseen log event) for embedding layer.
    # maybe not useful
    embedding.append(np.zeros(word_dims))
    embedding = np.asarray(embedding)

    plelog = PLELog(embedding, num_layer, lstm_hiddens, processor.label2id)

    log = 'layer={}_hidden={}_epoch={}'.format(num_layer, lstm_hiddens, epochs)
    best_model_file = os.path.join(output_model_dir, log + '_best.pt')
    last_model_file = os.path.join(output_model_dir, log + '_last.pt')
    if not os.path.exists(output_model_dir):
        os.makedirs(output_model_dir)

    if mode == 'train':
        # Dimension reduction if specified.
        transformer = None
        if reduce_dimension != -1:
            start_time = time.time()
            print("Start FastICA, target dimension: %d" % reduce_dimension)
            transformer = FastICA(n_components=reduce_dimension)
            train_reprs = transformer.fit_transform(train_reprs)
            for idx, inst in enumerate(train):
                inst.repr = train_reprs[idx]
            print('Finished at %.2f' % (time.time() - start_time))

        # Probabilistic labeling.
        labeled_train = label_generator.auto_label(train, normal_ids)

        # Below is used to test if the loaded result match the original clustering result.
        TP, TN, FP, FN = 0, 0, 0, 0
        for inst in labeled_train:
            if inst.predicted == 'Normal':
                if inst.label == 'Normal':
                    TN += 1
                else:
                    FN += 1
            else:
                if inst.label == 'Anomalous':
                    TP += 1
                else:
                    FP += 1
        plelog.logger.info('HDBSCAN results on train: TP %d TN %d FP %d FN %d' % (TP, TN, FP, FN))
        p, r, f = get_precision_recall(TP, TN, FP, FN)
        plelog.logger.info('%.4f, %.4f, %.4f' % (p, r, f))

        # Train
        optimizer = Optimizer(filter(lambda p: p.requires_grad, plelog.model.parameters()))
        bestClassifier = None
        global_step = 0
        bestF = 0
        batch_num = int(np.ceil(len(labeled_train) / float(batch_size)))

        for epoch in range(epochs):
            plelog.model.train()
            start = time.strftime("%H:%M:%S")
            plelog.logger.info("Starting epoch: %d | phase: train | start time: %s | learning rate: %s" %
                               (epoch + 1, start, optimizer.lr))
            batch_iter = 0
            correct_num, total_num = 0, 0
            # start batch
            for onebatch in data_iter(labeled_train, batch_size, True):
                plelog.model.train()
                tinst = generate_tinsts_binary_label(onebatch, processor.tag2id)
                if device.type == 'cuda':
                    tinst.to_cuda(device)
                loss = plelog.forward(tinst.inputs, tinst.targets)
                loss_value = loss.data.cpu().numpy()
                loss.backward()
                if batch_iter % 100 == 0:
                    plelog.logger.info("Step:%d, Iter:%d, batch:%d, loss:%.2f" \
                                       % (global_step, epoch, batch_iter, loss_value))
                batch_iter += 1
                if batch_iter % 1 == 0 or batch_iter == batch_num:
                    nn.utils.clip_grad_norm_(
                        filter(lambda p: p.requires_grad, plelog.model.parameters()),
                        max_norm=1)
                    optimizer.step()
                    plelog.model.zero_grad()
                    global_step += 1
                if dev:
                    if batch_iter % 700 == 0 or batch_iter == batch_num:
                        plelog.logger.info('Testing on validation set.')
                        _, _, f = plelog.evaluate(dev)
                        if f > bestF:
                            plelog.logger.info("Exceed best f: history = %.2f, current = %.2f" % (bestF, f))
                            torch.save(plelog.model.state_dict(), best_model_file)
                            bestF = f
            plelog.logger.info('Training epoch %d finished.' % epoch)
            torch.save(plelog.model.state_dict(), last_model_file)

    thre = 0
    step = 0.1
    bestP, bestR, bestF = 0, 0, 0
    best_thre = -1
    while thre <= 1:
        plelog.logger.info('=== Last Model ===')
        plelog.model.load_state_dict(torch.load(last_model_file, map_location='cuda:0'))
        if best_thre == -1:
            bestP, bestR, bestF = plelog.evaluate(test, thre)
            best_thre = thre
        else:
            plelog.model.load_state_dict(torch.load(last_model_file, map_location='cuda:0'))
            p, r, f = plelog.evaluate(test, thre)
            if f > bestF:
                bestP = p
                bestR = r
                bestF = f
                best_thre = thre
        if os.path.exists(best_model_file):
            plelog.logger.info('=== Best Model ===')
            plelog.model.load_state_dict(torch.load(best_model_file, map_location='cuda:0'))
            p, r, f = plelog.evaluate(test, thre)
            if f > bestF:
                bestP = p
                bestR = r
                bestF = f
                best_thre = thre
        thre += step
    plelog.logger.info(
        'Best results are p %.6f r %.6f f %.6f, by the threshold of %.2f' % (bestP, bestR, bestF, best_thre))
