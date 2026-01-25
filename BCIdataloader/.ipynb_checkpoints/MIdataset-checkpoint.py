from torch.utils.data import Dataset
import torch
import numpy as np
import os

class EEGDataset(Dataset):
    def __init__(self, filenames, sample_keys =['inputs','attention_mask'], chunk_len=500, num_chunks=10, ovlp=50, root_path="", normalization=True, start_samp_pnt=-1):
        
        if root_path == "":

            self.filenames = filenames
        else:
            self.filenames = [root_path + fn for fn in filenames if os.path.isfile(os.path.join(root_path, fn))]
            self.root_path = root_path
            
        print("Number of subjects loaded: ", len(self.filenames))

        
        self.chunk_len = chunk_len #Length (in samples) of each temporal segment or "chunk"
        self.num_chunks = num_chunks #Number of chunks to extract from each EEG signal.
        self.ovlp = ovlp #Overlap (in samples) between consecutive chunks
        self.sample_keys = sample_keys #What information are going to be return 
        self.do_normalization = normalization # Whether to apply normalization for each channel 

    def __len__(self):
        """
        Return the number of EEG sample in the dataset
        """
        return len(self.filenames)

    def __getitem__(self, idx):
        """
        Returns one item of the dataset 
        """
        data = self.load_tensor(self.filenames[idx])
        #===reorder channels====
        data = self.reorder_channels(data)
        return self.preprocess_sample(data, seq_len=self.nlog_dirum_chunks)

    @staticmethod
    def _pad_seq_right_to_n(
        seq: np.ndarray,
        n: int,
        pad_value: float = 0
        ) -> np.ndarray:
        """
        Padding function:  ensuring that all samples in a batch have the same temporal dimension.
        """
        if n== seq.shape[0] : 
            return seq
        return np.concatenate([seq, np.ones(n-seq.shape[0],*seq.shape[1:])*pad_value], axis=0)
        
    def load_tensor(self, filename):
        # tensor_fn = filename[:-3] + 'pt'
        tensor_data = torch.load(filename)
        return tensor_data.numpy()

    def reorder_channels(self, data):
        """
        Reorders EEG channels according to a predefined montage (e.g., FP1, F3, CZ, O2…).
        Ensures consistent channel order across subjects and datasets
        """
        chann_labels = {'FP1': 0, 'FP2': 1, 'F3': 2, 'F4': 3, 'C3': 4, 'C4': 5, 'P3': 6, 'P4': 7, 'O1': 8, 'O2': 9, 'F7': 10, 'F8': 11, 'T3': 12, 'T4': 13, 'T5': 14, 'T6': 15, 'FZ': 16, 'CZ': 17, 'PZ': 18, 'OZ': 19, 'T1': 20, 'T2': 21}
        reorder_labels = {'FP1': 0, 'FP2': 1, 'F7': 2, 'F3': 3, 'FZ': 4, 'F4': 5, 'F8': 6, 'T1': 7, 'T3': 8, 'C3': 9, 'CZ': 10, 'C4': 11, 'T4': 12, 'T2': 13, 'T5': 14, 'P3': 15, 'PZ': 16, 'P4': 17, 'T6': 18, 'O1': 19, 'OZ': 20, 'O2': 21}

        reordered = np.zeros_like(data)
        for label, target_idx in reorder_labels.items():
            mapped_idx = chann_labels[label]
            reordered[target_idx, :] = data[mapped_idx, :]
        
        return reordered

    def split_chunks(self, data, length=500, ovlp=50, num_chunks=10, start_point=-1): 

        """
        Splits continuous EEG data into overlapping temporal segments (windows).
        Each chunk represents a short EEG segment used as a model input window.
        
        Maintains temporal overlap to preserve continuity
        
        Allows random start points to add variability during training
        """

        all_chunks = []
        total_len = data.shape[1]
        actual_num_chunks = num_chunks
        
        if start_point == -1:
            if num_chunks * length > total_len - 1:
                start_point = 0
                actual_num_chunks = total_len // length
            else:
                start_point = np.random.randint(0, total_len - num_chunks * length)
        
        for i in range(actual_num_chunks):
            chunk = data[:, start_point: start_point + length]
            all_chunks.append(np.array(chunk))
            start_point = start_point + length - ovlp
        return np.array(all_chunks), start_point
    
    def normalize(self, data):
        """
        Applies z-score normalization per channel
        """
        mean = np.mean(data, axis=-1, keepdims=True)
        std = np.std(data, axis=-1, keepdims=True)
        # Ensure std is not zero to avoid division by zero.
        # If std is zero, normalization doesn't make sense, 
        # so you might set std to a small positive value or handle it in another way.
        # std = np.where(std == 0, 1e-23, std)
        return (data - mean) / (std + 1e-25)

    def preprocess_sample(
        self,
        sample,
        seq_len,
        labels=None
        ) :

        """
        Central function that performs all preprocessing steps:

        Normalize the EEG channels
        
        Chunk the signal into overlapping segments
        
        Pad sequences and attention masks
        
        Optionally flatten chunks for GPT-like sequential input
         
        """
        out = {}
        if self.do_normalization:
            sample = self.normalize(sample)

        chunks, seq_on = self.split_chunks(sample, self.chunk_len, self.ovlp, seq_len)
        attention_mask = np.ones(seq_len)
        chunks = self._pad_seq_right_to_n(
            seq=chunks,
            n=seq_len,
            pad_value=0
        )

        attention_mask = self._pad_seq_right_to_n(
            seq=attention_mask, 
            n=seq_len,
            pad_value=0
        )
        
        out["inputs"] = torch.from_numpy(chunks).to(torch.float)
        out["attention_mask"] = torch.from_numpy(attention_mask).to(torch.long)
        out['seq_on'] = seq_on
        out['seq_len'] = seq_len
        
        if self.sample_keys is not None:
            out = {
                key: out[key] 
                for key in self.sample_keys
                if key in out
            }

        if labels is not None:
            out['labels'] = torch.from_numpy(np.array(labels)).to(torch.long)
   
        return out

class MotorImageryDataset(EEGDataset):
    def __init__(self, filenames, sample_keys= ['inputs','attention_mask'], chunk_len=500, num_chunks=2, ovlp=0, root_path="", in_notebook  = False):
        super().__init__(filenames, sample_keys, chunk_len, num_chunks, ovlp, root_path=root_path)

        self.data_all = []

        for fn in self.filenames:
            self.data_all.append(np.load(fn))

        self.mi_types = {769: 'left', 770: 'right'}
        self.labels_string2int = {'left': 0, 'right': 1}

        self.Fs = 250  # Sampling frequency 250Hz from original paper

        self.P = np.load(os.path.join("/".join(root_path.split('/')[:-2]),'NeuroGPT_mini/tMatrix_value.npy')) # Projection matrixi to align the dataset channels with model inputs

        self.trials, self.labels, self.num_trials_per_sub = self.get_trials_all() # Raw data , labels and number of trials per subjects
        # keys of data ['s', 'etyp', 'epos', 'edur', 'artifacts']

    def __len__(self):
        return sum(self.num_trials_per_sub)

    def __getitem__(self, idx):
        return self.preprocess_sample(self.trials[idx], self.num_chunks, self.labels[idx])
        
    def get_trials_from_single_subj(self, sub_id):

            """
            Return the preprocessed EEG data and the corresponding labels for a given subject
            """
            raw = self.data_all[sub_id]['s'].T
            events_type = self.data_all[sub_id]['etyp'].T
            events_position = self.data_all[sub_id]['epos'].T
            events_duration = self.data_all[sub_id]['edur'].T
            artifacts = self.data_all[sub_id]['artifacts'].T
            # Channel default is C3
            startrial_code = 768
            starttrial_events = events_type == startrial_code
            idxs = [i for i, x in enumerate(starttrial_events[0]) if x]
    
            trial_labels = self.get_labels(sub_id)
    
            trials = []
            classes = []
            for j, index in enumerate(idxs):
                try:
                    if j >= len(trial_labels):
                        continue
            
                    classes.append(trial_labels[j])
            
                    start = events_position[0, index]
                    stop = start + events_duration[0, index]
                    trial = raw[:22, start+500 : stop-375]
                    trials.append(trial)
            
                except:
                    continue

            return trials, classes
    def map2pret(self, data):
        """
        Applies a projection matrix P to align channel configurations with the pretrained NeuroGPT model:
        """
        return np.matmul(self.P, data) # 22x22, 22xTime
        
    def get_labels(self, sub_id):
        events_type = self.data_all[sub_id]['etyp'].T
        trial_labels = [
            self.labels_string2int[self.mi_types[event]]
            for event in events_type[0]
            if event in self.mi_types
        ]
        return np.array(trial_labels)

        
    def get_trials_all(self):
        """
        Return the EEG preprocessed data, the labels for each subject 
        """
        trials_all = []
        labels_all = []
        total_num = []
        for sub_id in range(len(self.data_all)):
            trials, labels = self.get_trials_from_single_subj(sub_id)
            if len(trials) == 0:
                continue
            if len(labels) == 0:
                continue
            total_num.append(len(trials))
            
            trials_all.append(np.array(trials))
            labels_all.append(np.array(labels))
        # reordered_data = self.reorder_channels(np.vstack(trials_all))
        trials_all_arr = np.vstack(trials_all)
      
        # map to same channel configuration as pretraining
        trials_all_arr = self.map2pret(trials_all_arr)
        return self.normalize(trials_all_arr), np.array(labels_all).flatten(), total_num
    

   