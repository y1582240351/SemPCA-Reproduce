import sys

sys.path.extend([".", ".."])
from CONSTANTS import *
import argparse
from representations.templates.statistics import Simple_template_TF_IDF
from representations.sequences.statistics import FeatureExtractor
from preprocessing.preprocess import Preprocessor
from preprocessing.datacutter.SimpleCutting import cut_by_613
from module.Vocab import Vocab
from utils.common import summarize_subsequences, generate_subseq_dual_tinsts, data_iter, update_instances
from module.Optimizer import Optimizer
from models.lstm import LogAnomaly

if __name__ == '__main__':

    argparser = argparse.ArgumentParser()
    argparser.add_argument('--dataset', default='HDFS', type=str, help='Target dataset. Default HDFS')
    argparser.add_argument('--mode', default='train', type=str, help='train or test')
    argparser.add_argument('--parser', default='IBM', type=str,
                           help='Select parser, please see parser list for detail. Default Official.')
    argparser.add_argument('--num_layers', default=2, type=int)
    argparser.add_argument('--hidden_size', default=128, type=int)
    argparser.add_argument('--window_size', default=10, type=int)
    argparser.add_argument('--num_candidates', default=None, type=int)
    args, extra_args = argparser.parse_known_args()

    num_layers = args.num_layers
    hidden_size = args.hidden_size
    window_size = args.window_size
    dataset = args.dataset
    parser = args.parser
    mode = args.mode

    num_candidates = args.num_candidates

    # Hyper parameters.
    epochs = 5
    batch_size = 2048

    save_dir = os.path.join(PROJECT_ROOT, 'outputs')
    output_model_dir = os.path.join(save_dir, 'models/loganomaly/' + dataset + '_' + parser + '/model')
    output_res_dir = os.path.join(save_dir, 'results/loganomaly/' + dataset + '_' + parser + '/detect_res')

    if not os.path.exists(output_model_dir):
        os.makedirs(output_model_dir)
    if not os.path.exists(output_res_dir):
        os.makedirs(output_res_dir)

    processor = Preprocessor()

    # Training, Validating and Testing instances.
    template_encoder = Simple_template_TF_IDF()

    train, dev, test = processor.process(dataset=dataset, parsing=parser, template_encoding=template_encoder.present,
                                         cut_func=cut_by_613)

    # Construct Vocab
    vocab = Vocab()
    vocab.load_from_dict(processor.embedding)
    train, test, _ = update_instances(train, test)
    feature_extractor = FeatureExtractor()

    # Summarize number of predicted classes.
    num_classes = len(processor.train_event2idx)

    loganomaly = LogAnomaly(vocab, hidden_size, num_classes, device)

    loganomaly.logger.info(
        'Model hyperparameters: number of classes: {}, number of epochs: {}, batch size: {}'.format(num_classes,
                                                                                                    epochs,
                                                                                                    batch_size))

    log = 'layer={}_hidden={}_window={}_epoch={}_num_class={}'.format(num_layers, hidden_size, window_size, epochs,
                                                                      num_classes)
    best_model_output = output_model_dir + '/' + log + '_best.pt'
    last_model_output = output_model_dir + '/' + log + '.pt'

    # Use only normal instances for training.
    feature_extractor.fit_transform(np.asarray([np.asarray(inst.sequence, dtype=object) for inst in train]))
    train = list(filter(lambda x: x.label == 'Normal', train))
    # Randomly sample 50% of the normal instances for training.
    # train = random.sample(train, int(0.5 * len(train)))

    if mode == 'train':
        instances = summarize_subsequences(train, 10, 1)
        # Train
        optimizer = Optimizer(filter(lambda p: p.requires_grad, loganomaly.model.parameters()))
        bestClassifier = None
        global_step = 0
        bestF = 0
        batch_num = int(np.ceil(len(instances) / float(batch_size)))

        for epoch in range(epochs):
            loganomaly.model.train()
            start = time.strftime("%H:%M:%S")
            loganomaly.logger.info("Starting epoch: %d | phase: train | start time: %s | learning rate: %s" %
                                   (epoch + 1, start, optimizer.lr))
            batch_iter = 0
            # start batch
            for onebatch in data_iter(instances, batch_size, True):
                loganomaly.model.train()
                # Generate quantity_patterns
                subseq_sequential = []
                for subseq_inst in onebatch:
                    subseq_sequential.append(subseq_inst.sequential)
                subseq_sequential = np.asarray(subseq_sequential, dtype=object)
                subseq_quantities = feature_extractor.transform(subseq_sequential, silent=True)
                assert len(onebatch) == subseq_quantities.shape[0]
                for i in range(subseq_quantities.shape[0]):
                    onebatch[i].quantity = subseq_quantities[i, :]

                tinst = generate_subseq_dual_tinsts(onebatch, vocab, feature_extractor)
                if device.type == 'cuda':
                    tinst.to_cuda(device)
                loss = loganomaly.forward(tinst.inputs, tinst.targets)
                loss_value = loss.data.cpu().numpy()
                loss.backward()
                if batch_iter % 100 == 0:
                    loganomaly.logger.info("Step:%d, Iter:%d, batch:%d, loss:%.2f" \
                                           % (global_step, epoch, batch_iter, loss_value))
                batch_iter += 1
                if batch_iter % 1 == 0 or batch_iter == batch_num:
                    nn.utils.clip_grad_norm_(
                        filter(lambda p: p.requires_grad, loganomaly.model.parameters()),
                        max_norm=1)
                    optimizer.step()
                    loganomaly.model.zero_grad()
                    global_step += 1
            loganomaly.logger.info('Training epoch %d finished.' % epoch)
            torch.save(loganomaly.model.state_dict(), last_model_output)
        loganomaly.logger.info("Finish all training epochs.")
        pass
    loganomaly.logger.info('=== Final Model ===')
    loganomaly.model.load_state_dict(torch.load(last_model_output, map_location=device))
    loganomaly.evaluate(test, feature_extractor, num_candidates)
