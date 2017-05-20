import numpy as np
import shogi

from dlshogi.policy_network import *

class ProbabilisticPolicyPlayer(object):
    """A player that samples a move in proportion to the probability given by the
       policy.
       By manipulating the 'temperature', moves can be pushed towards totally random
       (high temperature) or towards greedy play (low temperature)
    """

    def __init__(self, model, temperature=0.67,
                 move_limit=255, greedy_start=None):
        assert(temperature > 0.0)
        self.model = model
        self.move_limit = move_limit
        self.beta = 1.0 / temperature
        self.greedy_start = greedy_start

    # Boltzmann distribution
    # see: Reinforcement Learning: An Introduction 2.3. SOFTMAX ACTION SELECTION
    def softmax_tempature(self, log_probabilities):
        # apply beta exponent to probabilities (in log space)
        log_probabilities = log_probabilities * self.beta
        # scale probabilities to a more numerically stable range (in log space)
        log_probabilities = log_probabilities - log_probabilities.max()
        # convert back from log space
        probabilities = np.exp(log_probabilities)
        # normalize the distribution
        return probabilities / probabilities.sum()

    def _select_moves(self, nn_output, state, moves):
        """helper function to normalize a distribution over the given list of moves
        and return a list of (move, prob) tuples
        """
        if len(moves) == 0:
            return []
        # convert Move to policy output label
        move_labels = [make_output_label(state, m) for m in moves]
        # get network activations at legal move locations
        log_probabilities = nn_output[move_labels]
        return zip(moves, move_labels, log_probabilities)
    
    def batch_eval_state(self, states, moves_lists):
        """Given a list of states, evaluates them all at once to make best use of GPU
        batching capabilities.
        Analogous to [eval_state(s) for s in states]
        Returns: a parallel list of move distributions as in eval_state
        """
        n_states = len(states)
        if n_states == 0:
            return []

        # make batch input data
        features1_batch = []
        features2_batch = []
        for s in states:
            features1, features2 = make_input_features_from_board(s)
            features1_batch.append(features1)
            features2_batch.append(features2)

        # pass all input through the network at once (backend makes use of
        # batches if len(states) is large)
        x1 = Variable(cuda.to_gpu(np.array(features1_batch, dtype=np.float32)))
        x2 = Variable(cuda.to_gpu(np.array(features2_batch, dtype=np.float32)))
        y = self.model(x1, x2, test=True)
        network_output = cuda.to_cpu(y.data)
        # default move lists to all legal moves
        results = [None] * n_states
        for i, st in enumerate(states):
            results[i] = self._select_moves(network_output[i], st, moves_lists[i])
        return results

    def get_moves(self, states):
        """Batch version of get_move. A list of moves is returned (one per state)
        """
        sensible_move_lists = [list(st.legal_moves) for st in states]
        all_moves_distributions = self.batch_eval_state(states, sensible_move_lists)
        move_list = [None] * len(states)
        label_list = [None] * len(states)
        for i, move_probs in enumerate(all_moves_distributions):
            if states[i].move_number > self.move_limit:
                continue
            else:
                if self.greedy_start is not None and len(states[i].history) >= self.greedy_start:
                    # greedy

                    max_prob = max(move_probs, key=itemgetter(2))
                    move_list[i] = max_prob[0]
                else:
                    # probabilistic

                    moves, labels, log_probabilities = zip(*move_probs)
                    # apply 'temperature' to the distribution
                    probabilities = self.softmax_tempature(np.array(log_probabilities))
                    # numpy interprets a list of tuples as 2D, so we must choose an
                    # _index_ of moves then apply it in 2 steps
                    choice_idx = np.random.choice(len(moves), p=probabilities)
                    move_list[i] = moves[choice_idx]
                    label_list[i] = labels[choice_idx]
        return move_list, label_list