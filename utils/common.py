from CONSTANTS import *
from sklearn.metrics import precision_recall_fscore_support
from torch.autograd import Variable

from entities.TensorInstances import TInstWithLogits, TensorInstance, DualTensorInstance, SequentialTensorInstance
from entities.instances import SubSequenceInstance

# Dispose Loggers.
CommonLogger = logging.getLogger('common')
CommonLogger.setLevel(logging.DEBUG)
console_handler = logging.StreamHandler(sys.stderr)
console_handler.setLevel(logging.DEBUG)
console_handler.setFormatter(
    logging.Formatter("%(asctime)s - %(name)s - " + SESSION + " - %(levelname)s: %(message)s"))

file_handler = logging.FileHandler(os.path.join(LOG_ROOT, 'LogRobust.log'))
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(
    logging.Formatter("%(asctime)s - %(name)s - " + SESSION + " - %(levelname)s: %(message)s"))

CommonLogger.addHandler(console_handler)
CommonLogger.addHandler(file_handler)
CommonLogger.info(
    'Construct logger for Common Methods succeeded, current working directory: %s, logs will be written in %s' %
    (os.getcwd(), LOG_ROOT))


def metrics(y_pred, y_true):
    """ Calucate evaluation metrics for precision, recall, and f1.

    Arguments
    ---------
        y_pred: ndarry, the predicted result list
        y_true: ndarray, the ground truth label list

    Returns
    -------
        precision: float, precision value
        recall: float, recall value
        f1: float, f1 measure value
    """
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary')
    return precision, recall, f1


def get_precision_recall(TP, TN, FP, FN):
    if TP == 0:
        return 0, 0, 0
    else:
        precision = TP / (TP + FP)
        recall = TP / (TP + FN)
        f = 2 * precision * recall / (precision + recall)
    return precision, recall, f


def not_empty(s):
    return s and s.strip()


def like_camel_to_tokens(camel_format):
    simple_format = []
    temp = ''
    flag = False

    if isinstance(camel_format, str):
        for i in range(len(camel_format)):
            if camel_format[i] == '-' or camel_format[i] == '_':
                simple_format.append(temp)
                temp = ''
                flag = False
            elif camel_format[i].isdigit():
                simple_format.append(temp)
                simple_format.append(camel_format[i])
                temp = ''
                flag = False
            elif camel_format[i].islower():
                if flag:
                    w = temp[-1]
                    temp = temp[:-1]
                    simple_format.append(temp)
                    temp = w + camel_format[i].lower()
                else:
                    temp += camel_format[i]
                flag = False
            else:
                if not flag:
                    simple_format.append(temp)
                    temp = ''
                temp += camel_format[i].lower()
                flag = True  # 需要回退
            if i == len(camel_format) - 1:
                simple_format.append(temp)
        simple_format = list(filter(not_empty, simple_format))
    return simple_format


def generate_inputs_and_labels(insts):
    inputs = []
    labels = np.zeros(len(insts))
    for idx, inst in enumerate(insts):
        inputs.append([int(x) for x in inst.sequence])
        if inst.label in ['Normal', 'Anomalous']:
            if inst.label == 'Normal':
                label = 0
            else:
                label = 1
        else:
            label = int(inst.label)

        labels[idx] = label
    return inputs, labels


def batch_slice(data, batch_size):
    batch_num = int(np.ceil(len(data) / float(batch_size)))
    for i in range(batch_num):
        cur_batch_size = batch_size if i < batch_num - 1 else len(data) - batch_size * i
        insts = [data[i * batch_size + b] for b in range(cur_batch_size)]
        yield insts


def data_iter(data, batch_size, shuffle=True):
    batched_data = []
    if shuffle: np.random.shuffle(data)
    batched_data.extend(list(batch_slice(data, batch_size)))
    if shuffle: np.random.shuffle(batched_data)
    for batch in batched_data:
        yield batch


def generate_tinsts(batched_insts, vocab):
    batch_size = len(batched_insts)
    # Summarize max length within this batch.
    slen = 0
    for inst in batched_insts:
        cur_len = len(inst.sequence)
        if cur_len > slen:
            slen = cur_len
    # Generate tensor instance.
    tinst = TensorInstance(batch_size, slen)
    b = 0
    for inst in batched_insts:
        tinst.src_ids.append(str(inst.id))
        tinst.g_truth[b] = vocab.tag2id(inst.label)
        cur_slen = len(inst.sequence)
        tinst.word_len[b] = cur_slen
        for index in range(cur_slen):
            tinst.src_words[b, index] = vocab.word2id(inst.sequence[index])
            tinst.mask[b, index] = 1
            pass
        b += 1
    return tinst


def generate_subseq_dual_tinsts(batched_insts, vocab, feature_extractor):
    batch_size = len(batched_insts)
    if batch_size == 0:
        print('Empty')
    # Summarize max length within this batch.
    slen = 0
    for inst in batched_insts:
        cur_len = len(inst.sequential)
        if cur_len > slen:
            slen = cur_len
    # Generate tensor instance.
    tinst = DualTensorInstance(batch_size, slen, len(feature_extractor.events))
    b = 0
    for inst in batched_insts:
        tinst.g_truth[b] = inst.label
        cur_slen = len(inst.sequential)
        tinst.word_len[b] = cur_slen
        for index in range(cur_slen):
            tinst.sequential[b, index] = vocab.word2id(inst.sequential[index])
            tinst.mask[b, index] = 1
            pass
        for dim in range(inst.quantity.shape[0]):
            tinst.quantity[b, dim] = inst.quantity[dim]
            pass
        b += 1
    return tinst


def generate_subseq_tinsts(batched_insts):
    batch_size = len(batched_insts)
    if batch_size == 0:
        print('Empty')
    # Summarize max length within this batch.
    slen = 0
    for inst in batched_insts:
        cur_len = len(inst.sequential)
        if cur_len > slen:
            slen = cur_len
    # Generate tensor instance.
    tinst = SequentialTensorInstance(batch_size, slen)
    b = 0
    for inst in batched_insts:
        tinst.g_truth[b] = inst.label
        cur_slen = len(inst.sequential)
        tinst.word_len[b] = cur_slen
        for index in range(cur_slen):
            tinst.src_words[b, index, 0] = inst.sequential[index]
            tinst.mask[b, index] = 1
            pass
        b += 1
    return tinst


def generate_tinsts_binary_label(batch_insts, tag2id, if_evaluate=False):
    slen = len(batch_insts[0].sequence)
    batch_size = len(batch_insts)
    for b in range(1, batch_size):
        cur_slen = len(batch_insts[b].sequence)
        if cur_slen > slen: slen = cur_slen
    tinst = TInstWithLogits(batch_size, slen, 2)
    b = 0
    for inst in batch_insts:
        tinst.src_ids.append(str(inst.id))
        confidence = 0.5 * inst.confidence
        if inst.predicted == '':
            inst.predicted = inst.label
        tinst.tags[b, tag2id[inst.predicted]] = 1 - confidence
        tinst.tags[b, 1 - tag2id[inst.predicted]] = confidence
        tinst.g_truth[b] = tag2id[inst.predicted]
        cur_slen = len(inst.sequence)
        tinst.word_len[b] = cur_slen
        for index in range(cur_slen):
            if index >= 500:
                break
            tinst.src_words[b, index] = inst.sequence[index]
            tinst.src_masks[b, index] = 1
        b += 1
    return tinst


def batch_variable_inst(insts, tagids, tag_logits, id2tag):
    if tag_logits is None:
        print('No prediction made, please check.')
        exit(-1)
    for inst, tagid, tag_logit in zip(insts, tagids, tag_logits):
        pred_label = id2tag[tagid]
        yield inst, pred_label == inst.label


def orthonormal_initializer(output_size, input_size):
    """
    adopted from Timothy Dozat https://github.com/tdozat/Parser/blob/master/lib/linalg.py
    """
    print(output_size, input_size)
    I = np.eye(output_size)
    lr = .1
    eps = .05 / (output_size + input_size)
    success = False
    tries = 0
    while not success and tries < 10:
        Q = np.random.randn(input_size, output_size) / np.sqrt(output_size)
        for i in range(100):
            QTQmI = Q.T.dot(Q) - I
            loss = np.sum(QTQmI ** 2 / 2)
            Q2 = Q ** 2
            Q -= lr * Q.dot(QTQmI) / (
                    np.abs(Q2 + Q2.sum(axis=0, keepdims=True) + Q2.sum(axis=1, keepdims=True) - 1) + eps)
            if np.max(Q) > 1e6 or loss > 1e6 or not np.isfinite(loss):
                tries += 1
                lr /= 2
                break
        success = True
    if success:
        print('Orthogonal pretrainer loss: %.2e' % loss)
    else:
        print('Orthogonal pretrainer failed, using non-orthogonal random matrix')
        Q = np.random.randn(input_size, output_size) / np.sqrt(output_size)
    return np.transpose(Q.astype(np.float32))


def drop_input_independent(word_embeddings, dropout_emb):
    batch_size, seq_length, _ = word_embeddings.size()
    word_masks = word_embeddings.data.new(batch_size, seq_length).fill_(1 - dropout_emb)
    word_masks = Variable(torch.bernoulli(word_masks), requires_grad=False)
    scale = 1.0 / (1.0 * word_masks + 1e-12)
    word_masks *= scale
    word_masks = word_masks.unsqueeze(dim=2)
    word_embeddings = word_embeddings * word_masks

    return word_embeddings


def update_sequences(instances, mapping):
    vocab_size = len(mapping)
    for inst in instances:
        processed_seq = []
        for event in inst.sequence:
            processed_seq.append(mapping[int(event)] if int(event) in mapping.keys() else vocab_size)
        inst.sequence.clear()
        inst.sequence = processed_seq


def summarize_subsequences(instances, window_size, step_size=1):
    '''
    Summarize subsequences for training, mainly used by LogAnomaly and DeepLog.
    Parameters
    ----------
    instances: Original log sequence instance in log data.
    window_size: Window size for subsequences.
    step_size: Step size when sliding the window.

    Returns
    -------
    Newly generated subsequence instances.
    '''
    new_instances = []
    # feature_extractor = FeatureExtractor()
    # train_inputs = []
    # for inst in instances:
    #     train_inputs.append(np.asarray(inst.sequence, dtype=int))
    # train_inputs = np.asarray(train_inputs, dtype=int)
    # feature_extractor.fit_transform(train_inputs)
    for inst in instances:
        seq_len = len(inst.sequence)
        i = 0
        if seq_len <= window_size:
            # num_append = window_size + 1 - seq_len
            # while num_append > 0:
            #     inst.sequence.append(-1)
            #     num_append -= 1
            # seq_len = len(inst.sequence)
            continue
        while True:
            if i + window_size >= seq_len:
                break
            subsequence = inst.sequence[i:i + window_size]
            np_subsequence = np.asarray(subsequence)
            if len(subsequence) == 0:
                print('Empty')
            new_inst = SubSequenceInstance(sequential=np_subsequence, label=inst.sequence[i + window_size])
            new_inst.belongs_to = inst.id
            new_instances.append(new_inst)
            i += step_size
            pass
    return new_instances


def update_instances(train=None, test=None):
    index = 0
    mapper = dict()
    for inst in train:
        for x in inst.sequence:
            x = int(x)
            if x not in mapper.keys():
                mapper[x] = index
                index += 1

    vocab_size = len(mapper.keys())

    for inst in train:
        processed_seq = []
        for event in inst.sequence:
            processed_seq.append(mapper[int(event)] if int(event) in mapper.keys() else vocab_size)
        inst.sequence.clear()
        inst.sequence = processed_seq

    for inst in test:
        processed_seq = []
        for event in inst.sequence:
            processed_seq.append(mapper[int(event)] if int(event) in mapper.keys() else vocab_size)
        inst.sequence.clear()
        inst.sequence = processed_seq
    return train, test, mapper
