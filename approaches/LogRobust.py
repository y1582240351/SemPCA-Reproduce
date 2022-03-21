import sys

sys.path.extend([".", ".."])
from CONSTANTS import *
import argparse
from representations.templates.statistics import Simple_template_TF_IDF
from preprocessing.preprocess import Preprocessor
from preprocessing.datacutter.SimpleCutting import cut_by_613
from module.Vocab import Vocab
from models.lstm import LogRobust
from module.Optimizer import Optimizer
from utils.common import data_iter, generate_tinsts

if __name__ == '__main__':

    argparser = argparse.ArgumentParser()
    argparser.add_argument('--dataset', default='Spirit', type=str, help='Target dataset. Default HDFS')
    argparser.add_argument('--mode', default='train', type=str, help='train or test')
    argparser.add_argument('--parser', default='IBM', type=str,
                           help='Select parser, please see parser list for detail. Default Official.')
    argparser.add_argument('--num_layers', default=2, type=int)
    argparser.add_argument('--hidden_size', default=128, type=int)
    args, extra_args = argparser.parse_known_args()

    num_layers = args.num_layers
    hidden_size = args.hidden_size
    dataset = args.dataset
    parser = args.parser
    mode = args.mode

    # Hyper parameters.
    epochs = 10
    batch_size = 256

    # Specify output folders.
    save_dir = os.path.join(PROJECT_ROOT, 'outputs')
    output_model_dir = os.path.join(save_dir, 'models/logrobust/' + dataset + '_' + parser + '/model')
    output_res_dir = os.path.join(save_dir, 'results/logrobust/' + dataset + '_' + parser + '/detect_res')

    # Create if not exist
    if not os.path.exists(output_model_dir):
        os.makedirs(output_model_dir)
    if not os.path.exists(output_res_dir):
        os.makedirs(output_res_dir)

    log = 'layer={}_hidden={}_epoch={}'.format(num_layers, hidden_size, epochs)
    best_model_output = output_model_dir + '/' + log + '_best.pt'
    last_model_output = output_model_dir + '/' + log + '.pt'

    # Prepare Training, Validating and Testing instances.
    processor = Preprocessor()
    template_encoder = Simple_template_TF_IDF()
    train, dev, test = processor.process(dataset=dataset, parsing=parser, template_encoding=template_encoder.present,
                                         cut_func=cut_by_613)

    # Prepare log template vocabulary
    vocab = Vocab()
    vocab.load_from_dict(processor.embedding)

    logrobust = LogRobust(vocab, hidden_size, num_layers, device)

    logrobust.logger.info(
        'Model hyperparameters: number of epochs: {}, batch size: {}'.format(epochs, batch_size))

    if mode == 'train':
        # Train
        optimizer = Optimizer(filter(lambda p: p.requires_grad, logrobust.model.parameters()))
        bestClassifier = None
        global_step = 0
        bestF = 0
        batch_num = int(np.ceil(len(train) / float(batch_size)))

        for epoch in range(epochs):
            logrobust.model.train()
            start = time.strftime("%H:%M:%S")
            logrobust.logger.info("Starting epoch: %d | phase: train | start time: %s | learning rate: %s" %
                                  (epoch + 1, start, optimizer.lr))
            batch_iter = 0
            # start batch
            for onebatch in data_iter(train, batch_size, True):
                logrobust.model.train()
                tinst = generate_tinsts(onebatch, vocab)
                if device.type == 'cuda':
                    tinst.to_cuda(device)
                loss = logrobust.forward(tinst.inputs, tinst.targets)
                loss_value = loss.data.cpu().numpy()
                loss.backward()
                if batch_iter % 100 == 0:
                    logrobust.logger.info("Step:%d, Iter:%d, batch:%d, loss:%.2f" \
                                          % (global_step, epoch, batch_iter, loss_value))
                batch_iter += 1
                if batch_iter % 1 == 0 or batch_iter == batch_num:
                    nn.utils.clip_grad_norm_(
                        filter(lambda p: p.requires_grad, logrobust.model.parameters()),
                        max_norm=1)
                    optimizer.step()
                    logrobust.model.zero_grad()
                    global_step += 1
                if dev:
                    if batch_iter % 700 == 0 or batch_iter == batch_num:
                        logrobust.logger.info('Testing on validation set.')
                        _, _, f = logrobust.evaluate(dev)
                        if f > bestF:
                            logrobust.logger.info("Exceed best f: history = %.2f, current = %.2f" % (bestF, f))
                            torch.save(logrobust.model.state_dict(), best_model_output)
                            bestF = f
            logrobust.logger.info('Training epoch %d finished.' % epoch)
            torch.save(logrobust.model.state_dict(), last_model_output)
        logrobust.logger.info("Finish all training epochs.")
    if os.path.exists(last_model_output):
        logrobust.logger.info('=== Final Model ===')
        logrobust.model.load_state_dict(torch.load(last_model_output))
        logrobust.evaluate(test)
    if os.path.exists(best_model_output):
        logrobust.logger.info('=== Best Model ===')
        logrobust.model.load_state_dict(torch.load(best_model_output))
        logrobust.evaluate(test)
