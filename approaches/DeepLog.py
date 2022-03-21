import sys

sys.path.extend([".", ".."])
from CONSTANTS import *
import argparse
from preprocessing.preprocess import Preprocessor
from preprocessing.datacutter.SimpleCutting import cut_by_613
from utils.common import update_sequences
from models.lstm import DeepLog

# Dispose Loggers.
DeepLogLogger = logging.getLogger('DeepLog')
DeepLogLogger.setLevel(logging.DEBUG)
console_handler = logging.StreamHandler(sys.stderr)
console_handler.setLevel(logging.DEBUG)
console_handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - " + SESSION + " - %(levelname)s: %(message)s"))

file_handler = logging.FileHandler(os.path.join(LOG_ROOT, 'DeepLog.log'))
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - " + SESSION + " - %(levelname)s: %(message)s"))

DeepLogLogger.addHandler(console_handler)
DeepLogLogger.addHandler(file_handler)
DeepLogLogger.info(
    'Construct logger for DeepLog succeeded, current working directory: %s, logs will be written in %s' %
    (os.getcwd(), LOG_ROOT))


def generate_inputs_by_instances(instances, window, step=1):
    '''
    Generate batched inputs by given instances.
    Parameters
    ----------
    instances: input insances for training.
    window: windows size for sliding window in DeepLog
    step: step size in DeepLog

    Returns: TensorDataset of training inputs and labels.
    -------

    '''
    num_sessions = 0
    inputs = []
    outputs = []
    for inst in instances:
        if inst.label == 'Normal':
            num_sessions += 1
            event_list = tuple(map(lambda n: n, map(int, inst.sequence)))
            for i in range(0, len(event_list) - window, step):
                inputs.append(event_list[i:i + window])
                outputs.append(event_list[i + window])
    DeepLogLogger.info('Number of sessions: {}'.format(num_sessions))
    DeepLogLogger.info('Number of seqs: {}'.format(len(inputs)))
    dataset = TensorDataset(torch.tensor(inputs, dtype=torch.long), torch.tensor(outputs, dtype=torch.long))
    return dataset


def evaluate(model, instances, num_candidates, out_file=None):
    '''
    Evaluate the effectiveness of model on instances. Return P, R and F-score.
    :param model: Trained model.
    :param instances: Testing set instances.
    :param num_candidates: Number of candidates (Top-n will be regarded as normal)
    :param out_file: If given, the evaluation result will be written into the file.
    :return: Precision, Recall and F1-Score.
    '''
    with torch.no_grad():
        TP = 0
        FP = 0
        TN = 0
        FN = 0
        normals = []
        anomalies = []
        normal_counter = {}
        anomalies_counter = {}
        for inst in instances:
            if inst.label == 'Normal':
                key = inst.seq_hash
                if key not in normal_counter.keys():
                    # If never seen, add a new instance for validating. Otherwise, only plus one on the count.
                    normals.append(inst)
                    normal_counter[key] = 0
                normal_counter[key] += 1
            else:
                key = inst.seq_hash
                if key not in anomalies_counter.keys():
                    anomalies.append(inst)
                    anomalies_counter[key] = 0
                anomalies_counter[key] += 1
        DeepLogLogger.info('Finish preparing validating set, start predicting:')
        for inst in tqdm(normals):
            predicted = False
            seq = inst.sequence + [-1] * (window_size + 1 - len(inst.sequence))
            log_sequence = list(map(lambda x: int(x), seq))
            for i in range(len(log_sequence) - window_size):
                seq = log_sequence[i:i + window_size]
                label = log_sequence[i + window_size]
                seq = torch.tensor(seq, dtype=torch.float).view(-1, window_size, input_size).to(device)
                label = torch.tensor(label).view(-1).to(device)
                output = model(seq)
                predict = torch.argsort(output, 1)[0][-num_candidates:]
                if label not in predict:
                    inst.predicted = "Anomalous"
                    predicted = True
                    FP += normal_counter[inst.seq_hash]
                    break
            if not predicted:
                inst.predicted = "Normal"
                TN += normal_counter[inst.seq_hash]
        FNCounter = Counter()
        for inst in tqdm(anomalies):
            predicted = False
            seq = inst.sequence + [-1] * (window_size + 1 - len(inst.sequence))
            log_sequence = list(map(lambda x: int(x), seq))
            for i in range(len(log_sequence) - window_size):
                seq = log_sequence[i:i + window_size]
                label = log_sequence[i + window_size]
                seq = torch.tensor(seq, dtype=torch.float).view(-1, window_size, input_size).to(device)
                label = torch.tensor(label).view(-1).to(device)
                output = model(seq)
                predict = torch.argsort(output, 1)[0][-num_candidates:]
                if label not in predict:
                    inst.predicted = "Anomalous"
                    predicted = True
                    TP += anomalies_counter[inst.seq_hash]
                    break
            if not predicted:
                inst.predicted = "Normal"
                FN += anomalies_counter[inst.seq_hash]

        if TP + FP != 0:
            P = 100 * TP / (TP + FP)
            R = 100 * TP / (TP + FN)
            F1 = 2 * P * R / (P + R)
        else:
            P = 0
            R = 0
            F1 = 0

        DeepLogLogger.info(
            'True positive(TP): {}, False Positive(FP): {} True negative(TN): {}, False Negative(FN): {} Precision: {:.3f}%, Recall: {:.3f}%, F1-measure: {:.3f}%'.format(
                TP, FP, TN, FN, P, R, F1))
        if out_file is not None:
            with open(out_file, 'w', encoding='utf-8') as writer:
                for inst in instances:
                    writer.write(str(inst) + '\n')
    return P, R, F1


if __name__ == '__main__':

    argparser = argparse.ArgumentParser()
    argparser.add_argument('--dataset', default='HDFS', type=str, help='Target dataset. Default HDFS')
    argparser.add_argument('--mode', default='train', type=str, help='train or test')
    argparser.add_argument('--parser', default='IBM', type=str,
                           help='Select parser, please see parser list for detail. Default Official.')
    argparser.add_argument('--num_layers', default=2, type=int)
    argparser.add_argument('--hidden_size', default=64, type=int)
    argparser.add_argument('--window_size', default=10, type=int)
    argparser.add_argument('--num_candidates', default=2, type=int)
    args, extra_args = argparser.parse_known_args()

    num_layers = args.num_layers
    hidden_size = args.hidden_size
    window_size = args.window_size
    dataset = args.dataset
    parser = args.parser
    mode = args.mode

    num_candidates = args.num_candidates

    num_epochs = 40
    input_size = 1
    batch_size = 512

    processor = Preprocessor()

    train, dev, test = processor.process(dataset=dataset, parsing=parser, template_encoding=None, cut_func=cut_by_613)
    num_classes = len(processor.train_event2idx)
    update_sequences(train, processor.train_event2idx)
    update_sequences(dev, processor.train_event2idx)
    update_sequences(test, processor.train_event2idx)

    DeepLogLogger.info(
        'Model hyperparameters: number of classes: {}, number of epochs: {}, batch size: {}'.format(num_classes,
                                                                                                    num_epochs,
                                                                                                    batch_size))
    save_dir = os.path.join(PROJECT_ROOT, 'outputs')
    output_model_dir = os.path.join(save_dir, 'models/deeplog/' + dataset + '_' + parser + '/model')
    output_res_dir = os.path.join(save_dir, 'results/deeplog/' + dataset + '_' + parser + '/detect_res')

    if not os.path.exists(output_model_dir):
        os.makedirs(output_model_dir)
    if not os.path.exists(output_res_dir):
        os.makedirs(output_res_dir)

    log = 'layer={}_hidden={}_window={}_epoch={}_num_class={}'.format(num_layers, hidden_size, window_size, num_epochs,
                                                                      num_classes)
    best_model_output = output_model_dir + '/' + log + '_best.pt'
    last_model_output = output_model_dir + '/' + log + '.pt'

    model = DeepLog(input_size, hidden_size, num_layers, num_classes).to(device)

    # Randomly sample 50% of normal log sequences for training.
    train = list(filter(lambda x: x.label == 'Normal', train))
    # train = random.sample(train, int(0.5 * len(train)))

    if mode == 'train':
        train_dataset = generate_inputs_by_instances(train, window=window_size)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)

        # Loss and optimizer
        criterion = nn.CrossEntropyLoss()
        # optimizer = Optimizer(filter(lambda p: p.requires_grad, model.parameters()), config)
        optimizer = torch.optim.Adam(model.parameters())
        # Train the model
        total_step = len(train_loader)
        start_time = time.time()
        for epoch in range(num_epochs):
            model.train()
            start = time.strftime("%H:%M:%S")
            DeepLogLogger.info("Starting epoch: %d | phase: train | start time: %s | learning rate: %f" %
                               (epoch + 1, start, optimizer.param_groups[0]['lr']))
            train_loss = 0
            for seq, label in tqdm(train_loader):
                # Forward pass
                seq = seq.clone().detach().view(-1, window_size, input_size).to(device)
                seq = seq.to(torch.float32)
                output = model(seq)
                loss = criterion(output, label.to(device))
                # Backward
                optimizer.zero_grad()
                loss.backward()
                train_loss += loss.item()
                optimizer.step()
            DeepLogLogger.info(
                'Epoch [{}/{}], train_loss: {:.4f}'.format(epoch + 1, num_epochs, train_loss / total_step))
            elapsed_time = time.time() - start_time
            DeepLogLogger.info('elapsed_time: {:.3f}s'.format(elapsed_time))
            torch.save(model.state_dict(), last_model_output)
        DeepLogLogger.info('Finished Training')

    elif mode == 'test':
        last_model = DeepLog(input_size, hidden_size, num_layers, num_classes)
        DeepLogLogger.info('Test last epoch\'s model by num_candidate=%d' % num_candidates)
        last_model.load_state_dict(torch.load(last_model_output))
        last_model.to(device)
        evaluate(last_model, test, num_candidates, os.path.join(output_res_dir, 'output_last.txt'))
        if os.path.exists(best_model_output):
            DeepLogLogger.info('Test by best dev\'s model')
            best_model = DeepLog(input_size, hidden_size, num_layers, num_classes)
            best_model.load_state_dict(torch.load(best_model_output))
            best_model.to(device)
            evaluate(best_model, test, num_candidates, os.path.join(output_res_dir, 'output_best.txt'))
    else:
        DeepLogLogger.error('Mode %s is not supported yet.' % mode)
        raise NotImplementedError
