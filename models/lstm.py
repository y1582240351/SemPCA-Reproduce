import tqdm

from CONSTANTS import *
from models.gru import AttGRUModel
from module.Attention import LinearAttention
from module.CPUEmbedding import CPUEmbedding
from module.Common import NonLinear
from utils.common import drop_input_independent, batch_variable_inst, generate_tinsts, generate_tinsts_binary_label
from utils.common import summarize_subsequences, generate_subseq_dual_tinsts, data_iter, generate_subseq_tinsts


class AttLSTMModel(nn.Module):
    # Dispose Loggers.
    _logger = logging.getLogger('AttLSTM')
    _logger.setLevel(logging.DEBUG)
    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(name)s - " + SESSION + " - %(levelname)s: %(message)s"))

    file_handler = logging.FileHandler(os.path.join(LOG_ROOT, 'AttLSTM.log'))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(name)s - " + SESSION + " - %(levelname)s: %(message)s"))

    _logger.addHandler(console_handler)
    _logger.addHandler(file_handler)
    _logger.info(
        'Construct logger for Attention-Based LSTM succeeded, current working directory: %s, logs will be written in %s' %
        (os.getcwd(), LOG_ROOT))

    @property
    def logger(self):
        return AttLSTMModel._logger

    def __init__(self, vocab, lstm_layers, lstm_hiddens, dropout, device):
        super(AttLSTMModel, self).__init__()
        self.dropout = dropout
        self.logger.info('==== Model Parameters ====')
        vocab_size, word_dims = vocab.vocab_size, vocab.word_dim
        self.word_embed = CPUEmbedding(vocab_size, word_dims, padding_idx=vocab_size - 1)
        self.word_embed.weight.data.copy_(torch.from_numpy(vocab.embeddings))
        self.word_embed.weight.requires_grad = False
        self.logger.info('Input Dimension: %d' % word_dims)
        self.logger.info('Hidden Size: %d' % lstm_hiddens)
        self.logger.info('Num Layers: %d' % lstm_layers)
        self.logger.info('Dropout %.3f' % dropout)
        self.rnn = nn.LSTM(input_size=word_dims, hidden_size=lstm_hiddens, num_layers=lstm_layers,
                           batch_first=True, bidirectional=True, dropout=dropout)

        self.sent_dim = 2 * lstm_hiddens
        self.atten_guide = Parameter(torch.Tensor(self.sent_dim))
        self.atten_guide.data.normal_(0, 1)
        self.atten = LinearAttention(tensor_1_dim=self.sent_dim, tensor_2_dim=self.sent_dim)
        self.proj = NonLinear(self.sent_dim, 2)
        self.device = device
        if self.device.type == 'cuda':
            self.cuda(self.device)

    def reset_word_embed_weight(self, vocab, pretrained_embedding):
        vocab_size, word_dims = pretrained_embedding.shape
        self.word_embed = CPUEmbedding(vocab.vocab_size, word_dims, padding_idx=vocab.PAD)
        self.word_embed.weight.data.copy_(torch.from_numpy(pretrained_embedding))
        self.word_embed.weight.requires_grad = False

    def forward(self, inputs):
        words, masks = inputs
        embed = self.word_embed(words)
        if self.training:
            embed = drop_input_independent(embed, self.dropout)
        embed = embed.to(self.device)
        batch_size = embed.size(0)
        atten_guide = torch.unsqueeze(self.atten_guide, dim=1).expand(-1, batch_size)
        atten_guide = atten_guide.transpose(1, 0)
        hiddens, state = self.rnn(embed)
        sent_probs = self.atten(atten_guide, hiddens, masks)
        batch_size, srclen, dim = hiddens.size()
        sent_probs = sent_probs.view(batch_size, srclen, -1)
        represents = hiddens * sent_probs
        represents = represents.sum(dim=1)
        outputs = self.proj(represents)
        return outputs


class Dual_LSTM(nn.Module):
    # Dispose Loggers.
    _logger = logging.getLogger('Dual_LSTM')
    _logger.setLevel(logging.DEBUG)
    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(name)s - " + SESSION + " - %(levelname)s: %(message)s"))

    file_handler = logging.FileHandler(os.path.join(LOG_ROOT, 'Dual_LSTM.log'))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(name)s - " + SESSION + " - %(levelname)s: %(message)s"))

    _logger.addHandler(console_handler)
    _logger.addHandler(file_handler)
    _logger.info(
        'Construct logger for Duality LSTM succeeded, current working directory: %s, logs will be written in %s' %
        (os.getcwd(), LOG_ROOT))

    @property
    def logger(self):
        return Dual_LSTM._logger

    def __init__(self, vocab, lstm_hiddens, num_classes, dropout, device):
        super(Dual_LSTM, self).__init__()
        self.dropout = dropout
        self.logger.info('==== Model Parameters ====')
        vocab_size, word_dims = vocab.vocab_size, vocab.word_dim
        self.word_embed = CPUEmbedding(vocab_size, word_dims, padding_idx=vocab_size - 1)
        self.word_embed.weight.data.copy_(torch.from_numpy(vocab.embeddings))
        self.word_embed.weight.requires_grad = False
        self.logger.info('Input Dimension: %d' % word_dims)
        self.logger.info('Hidden Size: %d' % lstm_hiddens)
        self.logger.info('Num Layers: %d' % 1)
        self.logger.info('Dropout %.3f' % dropout)
        self.sequential = nn.LSTM(input_size=word_dims, hidden_size=lstm_hiddens, num_layers=1,
                                  batch_first=True, bidirectional=False, dropout=dropout)
        self.quantitive = nn.LSTM(input_size=1, hidden_size=lstm_hiddens, num_layers=1,
                                  batch_first=True, bidirectional=False, dropout=dropout)
        self.sent_dim = 2 * lstm_hiddens
        self.num_classes = num_classes
        self.proj = NonLinear(self.sent_dim, num_classes)
        self.device = device
        # if self.device.type == 'cuda':
        #     self.cuda(self.device)

    def forward(self, inputs):
        sequential_pattern, quantity_pattern, masks = inputs
        embed = self.word_embed(sequential_pattern)
        if self.training:
            embed = drop_input_independent(embed, self.dropout)
        embed = embed.to(self.device)
        out_semantic, _ = self.sequential(embed)
        quantity_pattern = torch.unsqueeze(quantity_pattern, dim=2)
        out_quantity, (quantitiy_h_n, _) = self.quantitive(quantity_pattern)
        represents = torch.cat((out_semantic[:, -1, :], out_quantity[:, -1, :]), -1)
        outputs = self.proj(represents)
        return outputs


class LogAnomaly():
    # Dispose Loggers.
    _logger = logging.getLogger('LogAnomaly')
    _logger.setLevel(logging.DEBUG)
    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(name)s - " + SESSION + " - %(levelname)s: %(message)s"))

    file_handler = logging.FileHandler(os.path.join(LOG_ROOT, 'LogAnomaly.log'))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(name)s - " + SESSION + " - %(levelname)s: %(message)s"))

    _logger.addHandler(console_handler)
    _logger.addHandler(file_handler)
    _logger.info(
        'Construct logger for LogAnomaly succeeded, current working directory: %s, logs will be written in %s' %
        (os.getcwd(), LOG_ROOT))

    @property
    def logger(self):
        return LogAnomaly._logger

    def __init__(self, vocab, hidden, num_classes, device):
        super(LogAnomaly, self).__init__()
        self.vocab = vocab
        self.hidden_size = hidden
        self.batch_size = 128
        self.test_batch_size = 1
        self.device = device
        self.num_classes = num_classes
        self.model = Dual_LSTM(self.vocab, self.hidden_size, num_classes, 0.33, self.device)
        if self.device.type == 'cuda':
            self.model = self.model.cuda(self.device)
        self.loss = nn.CrossEntropyLoss()

    def forward(self, inputs, targets):
        tag_logits = self.model(inputs)
        loss = self.loss(tag_logits, targets)
        return loss

    def predict(self, inputs):
        with torch.no_grad():
            tag_logits = self.model(inputs)
            pred_tags = torch.argsort(tag_logits, dim=1, descending=True).detach().cpu().numpy()
        return pred_tags

    def evaluate(self, instances, feature_extractor, num_candidate):
        self.logger.info('Start evaluating')
        assert num_candidate is not None
        block_classifications = {}
        with torch.no_grad():
            self.model.eval()
            block_groundtruth = {}
            for inst in instances:
                block_groundtruth[inst.id] = inst.label
                if len(inst.sequence) <= 10:
                    block_classifications[inst.id] = 'Anomalous'
            # Summarize subsequence instances for testing.
            all_subseq_instances = summarize_subsequences(instances, 10, 1)
            # Quantity Patterns.
            train_inputs = []
            hashed_sequential = {}
            unique_subsequence_instances = []
            key2prediction = {}
            for inst in all_subseq_instances:
                key = hash(inst)
                if key not in hashed_sequential.keys():
                    train_inputs.append(inst.sequential)
                    hashed_sequential[key] = len(train_inputs) - 1
                    unique_subsequence_instances.append(inst)
                    key2prediction[key] = []
            quantity_patterns = feature_extractor.transform(np.asarray(train_inputs, dtype=object))
            self.logger.info('Summarized %d unique subsequence instances from %d subsequence instances.' % (
                quantity_patterns.shape[0], len(all_subseq_instances)))
            assert len(hashed_sequential) == quantity_patterns.shape[0]
            for subseq_inst in unique_subsequence_instances:
                key = hash(subseq_inst)
                subseq_inst.quantity = quantity_patterns[hashed_sequential[key], :]

            # Generate batched TensorInstance for subsequence instances.
            total_iters = math.ceil(len(unique_subsequence_instances) / self.batch_size)
            pbar = tqdm(total=total_iters)
            for onebatch in data_iter(unique_subsequence_instances, self.batch_size, True):
                tinst = generate_subseq_dual_tinsts(onebatch, self.vocab, feature_extractor)
                if device.type == 'cuda':
                    tinst.to_cuda(device)
                self.model.eval()
                pred_tags = self.predict(tinst.inputs)
                for i, sub_inst in enumerate(onebatch):
                    key2prediction[hash(sub_inst)] = pred_tags[i]
                pbar.update(1)
            pbar.close()

            self.logger.info('Updating classification results to all subsequence instances.')
            for sub_inst in tqdm(all_subseq_instances):
                key = hash(sub_inst)
                if key in key2prediction.keys():
                    sub_inst.predictions = key2prediction[key]
                else:
                    self.logger.warning('Missing prediction foe block id: %s, subseqeunce is %s' % (
                        str(sub_inst.belongs_to), '[' + ' '.join([str(x) for x in sub_inst.sequential]) + ']'))
            # Start Evaluation.
            self.logger.info('Evaluating by num_candidate: %d...' % num_candidate)

            # Make predictions based on num_candidates.
            for sub_inst in all_subseq_instances:
                candidates = sub_inst.predictions[:num_candidate]
                # if the true label is included in the predictions, consider it as normal.
                if sub_inst.label in candidates or self.similar(candidates, sub_inst.label, self.vocab):
                    # If been recorded, won't change the result if current subsequence is normal.
                    if sub_inst.belongs_to not in block_classifications.keys():
                        block_classifications[sub_inst.belongs_to] = 'Normal'
                        pass
                    pass
                # Otherwise, the block contains this subsequence should be labeled as anomalous.
                else:
                    block_classifications[sub_inst.belongs_to] = 'Anomalous'
                    pass
                pass
            precision, recall, f1 = self.evaluate_metrics(block_classifications, block_groundtruth)
        return precision, recall, f1

    def evaluate_metrics(self, block_classifications, block_groundtruth):
        # Calculate P R and F
        assert len(block_groundtruth) == len(block_classifications)
        TP, TN, FP, FN = 0, 0, 0, 0
        for block in block_groundtruth.keys():
            ground_truth = block_groundtruth[block]
            bmatch = block_classifications[block] == ground_truth
            if bmatch:
                if ground_truth == 'Normal':
                    TN += 1
                else:
                    TP += 1
            else:
                if ground_truth == 'Normal':
                    FP += 1
                else:
                    FN += 1
        self.logger.info('TP: %d, TN: %d, FN: %d, FP: %d' % (TP, TN, FN, FP))
        if TP + FP != 0:
            precision = TP / (TP + FP)
            recall = TP / (TP + FN)
            f = 2 * precision * recall / (precision + recall)
            self.logger.info('Precision = %d / %d = %.5f, Recall = %d / %d = %.5f F1 score = %.5f'
                             % (TP, (TP + FP), precision, TP, (TP + FN), recall, f))
        else:
            self.logger.info('Precision is 0 and therefore f is 0')
            precision, recall, f = 0, 0, 0
        return precision, recall, f

    def similar(self, candidates, label, vocab, threshold=0.5):
        flag = False
        # # Construct semantic matrix
        # candidates = np.asarray([vocab.embeddings[vocab.word2id(x)] for x in candidates], dtype=float)
        # target = np.asarray(vocab.embeddings[vocab.word2id(label)], dtype=float)
        #
        # # Similarity calculation
        # sims = []
        # candidate_norm = np.linalg.norm(candidates, axis=1)
        # target_norm = np.linalg.norm(target)
        # dot_corss = np.dot(target, candidates.T)
        # sims = dot_corss / (candidate_norm * target_norm)
        # sims[np.isneginf(sims)] = 0
        # sims = 0.5 + 0.5 * sims
        #
        # # If maximum similarity is greater than 0.5, consider it as similar(normal)
        # max_sim = np.amax(sims, axis=0)
        # if max_sim > threshold:
        #     flag = True
        return flag


# class DeepLog():
#     # Dispose Loggers.
#     _logger = logging.getLogger('DeepLog')
#     _logger.setLevel(logging.DEBUG)
#     console_handler = logging.StreamHandler(sys.stderr)
#     console_handler.setLevel(logging.DEBUG)
#     console_handler.setFormatter(
#         logging.Formatter("%(asctime)s - %(name)s - " + SESSION + " - %(levelname)s: %(message)s"))
#
#     file_handler = logging.FileHandler(os.path.join(LOG_ROOT, 'DeepLog.log'))
#     file_handler.setLevel(logging.INFO)
#     file_handler.setFormatter(
#         logging.Formatter("%(asctime)s - %(name)s - " + SESSION + " - %(levelname)s: %(message)s"))
#
#     _logger.addHandler(console_handler)
#     _logger.addHandler(file_handler)
#     _logger.info(
#         'Construct logger for DeepLog succeeded, current working directory: %s, logs will be written in %s' %
#         (os.getcwd(), LOG_ROOT))
#
#     @property
#     def logger(self):
#         return DeepLog._logger
#
#     def __init__(self, input_size, num_layers, hidden, num_classes, device):
#         super(DeepLog, self).__init__()
#         self.hidden_size = hidden
#         self.test_batch_size = 1
#         self.device = device
#         self.num_classes = num_classes
#         self.num_layers = num_layers
#         self.model = nn.LSTM(input_size=input_size, hidden_size=hidden, num_layers=num_layers, batch_first=True,
#                              bidirectional=False,
#                              dropout=0.33)
#         self.classifier = nn.Linear(self.hidden_size, num_classes)
#         self.loss = nn.CrossEntropyLoss()
#         if self.device.type == 'cuda':
#             self.model = self.model.cuda(self.device)
#             self.classifier = self.classifier.cuda(self.device)
#             self.loss.cuda(self.device)
#
#     def param_tune(self, instances, batch_size):
#         self.logger.info('Parameter tuning on validation set.')
#         pre_anomalous_blocks = []
#         with torch.no_grad():
#             self.model.eval()
#             block_groundtruth = {}
#             for inst in instances:
#                 block_groundtruth[inst.id] = inst.label
#                 if len(inst.sequence) <= 10:
#                     pre_anomalous_blocks.append(inst.id)
#
#             # Summarize subsequence instances for testing.
#             all_subseq_instances = summarize_subsequences(instances, 10, 1)
#             # Summarize unique subserquence instances to accelerate the testing.
#             unique_subsequence_instances = []
#             key2prediction = {}
#             for inst in all_subseq_instances:
#                 key = hash(inst)
#                 if key not in key2prediction.keys():
#                     unique_subsequence_instances.append(inst)
#                     key2prediction[key] = []
#
#             # Generate batched TensorInstance for subsequence instances.
#             total_iters = math.ceil(len(unique_subsequence_instances) / batch_size)
#             pbar = tqdm(total=total_iters)
#             for onebatch in data_iter(unique_subsequence_instances, batch_size, True):
#                 tinst = generate_subseq_tinsts(onebatch)
#                 if device.type == 'cuda':
#                     tinst.to_cuda(device)
#                 self.model.eval()
#                 pred_tags = self.predict(tinst.inputs)
#                 for i, sub_inst in enumerate(onebatch):
#                     key2prediction[hash(sub_inst)] = pred_tags[i]
#                 pbar.update(1)
#             pbar.close()
#
#             self.logger.info('Updating classification results to all subsequence instances.')
#             for sub_inst in tqdm(all_subseq_instances):
#                 key = hash(sub_inst)
#                 if key in key2prediction.keys():
#                     sub_inst.predictions = key2prediction[key]
#                 else:
#                     self.logger.warning('Missing prediction foe block id: %s, subseqeunce is %s' % (
#                         str(sub_inst.belongs_to), '[' + ' '.join([str(x) for x in sub_inst.sequential]) + ']'))
#             # Start Evaluation.
#             if True:
#                 self.logger.info('Start auto-parameter filtering.')
#                 previous_F = -1
#                 best_prec, best_recall, best_f1 = 0, 0, 0
#                 best_num_candidate = 0
#                 i = 0
#                 while i < self.num_classes:
#                     num_candidate = i + 1
#                     i += 1
#                     block_classifications = {}
#                     for id in pre_anomalous_blocks:
#                         block_classifications[id] = 'Anomalous'
#                     self.logger.info('Number of candidates: %d' % num_candidate)
#                     # Make predictions based on current num_candidates.
#                     for sub_inst in all_subseq_instances:
#                         candidates = sub_inst.predictions[:num_candidate]
#                         # if the true label is included in the predictions, consider it as normal.
#                         if sub_inst.label in candidates:
#                             # If been recorded, won't change the result if current subsequence is normal.
#                             if sub_inst.belongs_to not in block_classifications.keys():
#                                 block_classifications[sub_inst.belongs_to] = 'Normal'
#                                 pass
#                             pass
#                         # Otherwise, the block contains this subsequence should be labeled as anomalous.
#                         else:
#                             block_classifications[sub_inst.belongs_to] = 'Anomalous'
#                             pass
#                         pass
#                     precision, recall, f1 = self.evaluate_metrics(block_classifications, block_groundtruth)
#
#                     # If the performance start to decline, stop.
#                     # if previous_F != -1 and f1 < previous_F:
#                     #     break
#                     # Update the last num_candidate's F1 score.
#                     previous_F = f1
#                     # If current f1 score is better, update recorder.
#                     if f1 >= best_f1:
#                         best_prec = precision
#                         best_recall = recall
#                         best_f1 = f1
#                         best_num_candidate = num_candidate
#                     pass
#                 self.logger.info('Selected best num_candidate: %d, Precision: %.4f, Recall: %.4f, F1-Score: %.4f.' % (
#                     best_num_candidate, best_prec, best_recall, best_f1
#                 ))
#
#         return best_num_candidate
#
#     def forward(self, inputs, targets):
#         sequences, _ = inputs
#         h0 = torch.zeros(self.num_layers, sequences.size(0), self.hidden_size).to(self.device)
#         c0 = torch.zeros(self.num_layers, sequences.size(0), self.hidden_size).to(self.device)
#         out, _ = self.model(sequences, (h0, c0))
#         tag_logits = self.classifier(out[:, -1, :])
#         loss = self.loss(tag_logits, targets)
#         return loss
#
#     def predict(self, inputs):
#         sequences, _ = inputs
#         with torch.no_grad():
#             h0 = torch.zeros(self.num_layers, sequences.size(0), self.hidden_size).to(device)
#             c0 = torch.zeros(self.num_layers, sequences.size(0), self.hidden_size).to(device)
#             out, _ = self.model(sequences, (h0, c0))
#             tag_logits = self.classifier(out[:, -1, :])
#             pred_tags = torch.argsort(tag_logits, dim=1, descending=True).detach().cpu().numpy()
#         return pred_tags
#
#     def evaluate(self, instances, num_candidate, batch_size):
#         self.logger.info('Start evaluating')
#         assert num_candidate is not None
#         block_classifications = {}
#         with torch.no_grad():
#             self.model.eval()
#             block_groundtruth = {}
#             for inst in instances:
#                 block_groundtruth[inst.id] = inst.label
#                 if len(inst.sequence) <= 10:
#                     block_classifications[inst.id] = 'Anomalous'
#
#             # Summarize subsequence instances for testing.
#             all_subseq_instances = summarize_subsequences(instances, 10, 1)
#             # Summarize unique subsequence instances to accelerate the testing
#             unique_subsequence_instances = []
#             key2prediction = {}
#             for inst in all_subseq_instances:
#                 key = hash(inst)
#                 if key not in key2prediction.keys():
#                     unique_subsequence_instances.append(inst)
#                     key2prediction[key] = []
#
#             # Generate batched TensorInstance and predict for subsequence instances.
#             total_iters = math.ceil(len(unique_subsequence_instances) / batch_size)
#             pbar = tqdm(total=total_iters)
#             for onebatch in data_iter(unique_subsequence_instances, batch_size, True):
#                 tinst = generate_subseq_tinsts(onebatch)
#                 if device.type == 'cuda':
#                     tinst.to_cuda(device)
#                 self.model.eval()
#                 pred_tags = self.predict(tinst.inputs)
#                 for i, sub_inst in enumerate(onebatch):
#                     key2prediction[hash(sub_inst)] = pred_tags[i]
#                 pbar.update(1)
#             pbar.close()
#
#             self.logger.info('Updating classification results to all subsequence instances.')
#             for sub_inst in tqdm(all_subseq_instances):
#                 key = hash(sub_inst)
#                 if key in key2prediction.keys():
#                     sub_inst.predictions = key2prediction[key]
#                 else:
#                     self.logger.warning('Missing prediction foe block id: %s, subseqeunce is %s' % (
#                         str(sub_inst.belongs_to), '[' + ' '.join([str(x) for x in sub_inst.sequential]) + ']'))
#
#             # Start Evaluation.
#             self.logger.info('Evaluating by num_candidate: %d...' % num_candidate)
#             # Make predictions based on num_candidates.
#             for sub_inst in all_subseq_instances:
#                 candidates = sub_inst.predictions[:num_candidate]
#                 # if the true label is included in the predictions, consider it as normal.
#                 if sub_inst.label in candidates:
#                     # If been recorded, won't change the result if current subsequence is normal.
#                     if sub_inst.belongs_to not in block_classifications.keys():
#                         block_classifications[sub_inst.belongs_to] = 'Normal'
#                         pass
#                     pass
#                 # Otherwise, the block contains this subsequence should be labeled as anomalous.
#                 else:
#                     block_classifications[sub_inst.belongs_to] = 'Anomalous'
#                     pass
#                 pass
#             precision, recall, f1 = self.evaluate_metrics(block_classifications, block_groundtruth)
#         return precision, recall, f1
#
#     def evaluate_metrics(self, block_classifications, block_groundtruth):
#         # Calculate P R and F
#         assert len(block_groundtruth) == len(block_classifications)
#         TP, TN, FP, FN = 0, 0, 0, 0
#         for block in block_groundtruth.keys():
#             ground_truth = block_groundtruth[block]
#             bmatch = block_classifications[block] == ground_truth
#             if bmatch:
#                 if ground_truth == 'Normal':
#                     TN += 1
#                 else:
#                     TP += 1
#             else:
#                 if ground_truth == 'Normal':
#                     FP += 1
#                 else:
#                     FN += 1
#         self.logger.info('TP: %d, TN: %d, FN: %d, FP: %d' % (TP, TN, FN, FP))
#         if TP + FP != 0:
#             precision = TP / (TP + FP)
#             recall = TP / (TP + FN)
#             f = 2 * precision * recall / (precision + recall)
#             self.logger.info('Precision = %d / %d = %.5f, Recall = %d / %d = %.5f F1 score = %.5f'
#                              % (TP, (TP + FP), precision, TP, (TP + FN), recall, f))
#         else:
#             self.logger.info('Precision is 0 and therefore f is 0')
#             precision, recall, f = 0, 0, 0
#         return precision, recall, f

class DeepLog(nn.Module):
    def __init__(self, input_dim, hidden, layer, num_classes):
        super(DeepLog, self).__init__()
        self.hidden_size = hidden
        self.num_layers = layer
        self.input_size = input_dim
        self.lstm = nn.LSTM(self.input_size, self.hidden_size, self.num_layers, batch_first=True)
        self.fc = nn.Linear(self.hidden_size, num_classes)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out


class LogRobust:
    # Dispose Loggers.
    _logger = logging.getLogger('LogRobust')
    _logger.setLevel(logging.DEBUG)
    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(name)s - " + SESSION + " - %(levelname)s: %(message)s"))

    file_handler = logging.FileHandler(os.path.join(LOG_ROOT, 'LogRobust.log'))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(name)s - " + SESSION + " - %(levelname)s: %(message)s"))

    _logger.addHandler(console_handler)
    _logger.addHandler(file_handler)
    _logger.info(
        'Construct logger for LogRobust succeeded, current working directory: %s, logs will be written in %s' %
        (os.getcwd(), LOG_ROOT))

    @property
    def logger(self):
        return LogRobust._logger

    def __init__(self, vocab, hidden, layer, device):
        super(LogRobust, self).__init__()
        self.vocab = vocab
        self.id2tag = {0: 'Normal', 1: 'Anomalous'}
        self.hidden_size = hidden
        self.num_layers = layer
        self.input_size = vocab.word_dim
        self.batch_size = 128
        self.test_batch_size = 1024
        self.device = device
        self.model = AttLSTMModel(self.vocab, self.num_layers, self.hidden_size, 0.33, self.device)
        if device.type == 'cuda':
            self.model = self.model.cuda(device)
        self.loss = nn.CrossEntropyLoss()

    def forward(self, inputs, targets):
        tag_logits = self.model(inputs)
        loss = self.loss(tag_logits, targets)
        return loss

    def predict(self, inputs):
        with torch.no_grad():
            tag_logits = self.model(inputs)
            pred_tags = tag_logits.detach().max(1)[1].cpu().numpy()
        return pred_tags, tag_logits

    def evaluate(self, instances):
        self.logger.info('Start evaluating')
        with torch.no_grad():
            self.model.eval()
            globalBatchNum = 0
            TP, TN, FP, FN = 0, 0, 0, 0
            tag_correct, tag_total = 0, 0
            for onebatch in data_iter(instances, self.test_batch_size, False):
                tinst = generate_tinsts(onebatch, self.vocab)
                if device.type == 'cuda':
                    tinst.to_cuda(device)
                self.model.eval()
                pred_tags, tag_logits = self.predict(tinst.inputs)
                for inst, bmatch in batch_variable_inst(onebatch, pred_tags, tag_logits, self.id2tag):
                    tag_total += 1
                    if bmatch:
                        tag_correct += 1
                        if inst.label == 'Normal':
                            TN += 1
                        else:
                            TP += 1
                    else:
                        if inst.label == 'Normal':
                            FP += 1
                        else:
                            FN += 1
                globalBatchNum += 1
            self.logger.info('TP: %d, TN: %d, FN: %d, FP: %d' % (TP, TN, FN, FP))
            if TP + FP != 0:
                precision = 100 * TP / (TP + FP)
                recall = 100 * TP / (TP + FN)
                f = 2 * precision * recall / (precision + recall)
                end = time.time()
                self.logger.info('Precision = %d / %d = %.4f, Recall = %d / %d = %.4f F1 score = %.4f'
                                 % (TP, (TP + FP), precision, TP, (TP + FN), recall, f))
            else:
                self.logger.info('Precision is 0 and therefore f is 0.')
                precision, recall, f = 0, 0, 0
        return precision, recall, f


class PLELog:
    def __init__(self, embedding, num_layer, hidden_size, label2id):
        assert isinstance(embedding, np.ndarray)
        self.label2id = label2id
        self.tag2id = {'Normal': 0, 'Anomalous': 1}
        self.id2tag = {0: 'Normal', 1: 'Anomalous'}
        self.logger = self.create_logger()
        self.embedding = embedding
        self.num_layer = num_layer
        self.hidden_size = hidden_size
        self.batch_size = 100
        self.test_batch_size = 1024
        self.model = AttGRUModel(self.embedding, self.num_layer, self.hidden_size, dropout=0.33)
        if torch.cuda.is_available():
            self.model = self.model.cuda(device)
        self.loss = nn.BCELoss()

    def create_logger(self):
        # Dispose Loggers.
        PLELogLogger = logging.getLogger('PLELog')
        PLELogLogger.setLevel(logging.DEBUG)
        console_handler = logging.StreamHandler(sys.stderr)
        console_handler.setLevel(logging.DEBUG)
        console_handler.setFormatter(
            logging.Formatter("%(asctime)s - %(name)s - " + SESSION + " - %(levelname)s: %(message)s"))
        file_handler = logging.FileHandler(os.path.join(LOG_ROOT, 'PLELog.log'))
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s - %(name)s - " + SESSION + " - %(levelname)s: %(message)s"))
        PLELogLogger.addHandler(console_handler)
        PLELogLogger.addHandler(file_handler)
        PLELogLogger.info(
            'Construct logger for PLELog succeeded, current working directory: %s, logs will be written in %s' %
            (os.getcwd(), LOG_ROOT))
        return PLELogLogger

    def forward(self, inputs, targets):
        tag_logits = self.model(inputs)
        tag_logits = F.softmax(tag_logits, dim=1)
        loss = self.loss(tag_logits, targets)
        return loss

    def predict(self, inputs, threshold=None):
        with torch.no_grad():
            tag_logits = self.model(inputs)
            tag_logits = F.softmax(tag_logits)
        if threshold is not None:
            probs = tag_logits.detach().cpu().numpy()
            anomaly_id = self.label2id['Anomalous']
            pred_tags = np.zeros(probs.shape[0])
            for i, logits in enumerate(probs):
                if logits[anomaly_id] >= threshold:
                    pred_tags[i] = anomaly_id
                else:
                    pred_tags[i] = 1 - anomaly_id

        else:
            pred_tags = tag_logits.detach().max(1)[1].cpu()
        return pred_tags, tag_logits

    def evaluate(self, instances, threshold=0.5):
        self.logger.info('Start evaluating by threshold %.3f' % threshold)
        with torch.no_grad():
            self.model.eval()
            globalBatchNum = 0
            TP, TN, FP, FN = 0, 0, 0, 0
            tag_correct, tag_total = 0, 0
            for onebatch in data_iter(instances, self.test_batch_size, False):
                tinst = generate_tinsts_binary_label(onebatch, self.tag2id, True)
                if device.type == 'cuda':
                    tinst.to_cuda(device)
                self.model.eval()
                pred_tags, tag_logits = self.predict(tinst.inputs, threshold)
                for inst, bmatch in batch_variable_inst(onebatch, pred_tags, tag_logits, self.id2tag):
                    tag_total += 1
                    if bmatch:
                        tag_correct += 1
                        if inst.label == 'Normal':
                            TN += 1
                        else:
                            TP += 1
                    else:
                        if inst.label == 'Normal':
                            FP += 1
                        else:
                            FN += 1
                globalBatchNum += 1
            self.logger.info('TP: %d, TN: %d, FN: %d, FP: %d' % (TP, TN, FN, FP))
            if TP != 0:
                precision = 100 * TP / (TP + FP)
                recall = 100 * TP / (TP + FN)
                f = 2 * precision * recall / (precision + recall)
                end = time.time()
                self.logger.info('Precision = %d / %d = %.4f, Recall = %d / %d = %.4f F1 score = %.4f'
                                 % (TP, (TP + FP), precision, TP, (TP + FN), recall, f))
            else:
                self.logger.info('Precision is 0 and therefore f is 0')
                precision, recall, f = 0, 0, 0
        return precision, recall, f
