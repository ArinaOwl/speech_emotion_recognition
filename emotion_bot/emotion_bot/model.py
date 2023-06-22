import torch
import torch.nn as nn
import numpy as np
from typing import List, Optional, Union, Tuple
import librosa
from transformers import SequenceFeatureExtractor, BatchFeature, TensorType


class FeatureExtractor(SequenceFeatureExtractor):

    model_input_names = ["input_values", "attention_mask"]

    def __init__(
            self,
            feature_size=1,
            sampling_rate=16000,
            n_harmonics=4,
            max_length=1024,
            padding_value=0.0,
            do_normalize=True,
            return_attention_mask=False,
            **kwargs,
    ):
        super().__init__(feature_size=feature_size, sampling_rate=sampling_rate, padding_value=padding_value, **kwargs)
        self.n_harmonics = n_harmonics
        self.max_length = max_length
        self.do_normalize = do_normalize
        self.return_attention_mask = return_attention_mask

    def _extract_f0_harmonic_features(
            self,
            waveform: np.ndarray,
            sampling_rate: int,
            max_length: int,
            n_harm: int,
    ) -> np.ndarray:
        """
        Get f0 and harmonic features using librosa.
        The waveform should not be normalized before feature extraction.
        """
        f0, voicing, voicing_p = librosa.pyin(waveform, sr=sampling_rate,
                                              fmin=64., fmax=1046.)
        s = np.abs(librosa.stft(waveform))
        freqs = librosa.fft_frequencies(sr=sampling_rate)
        f0_harm = librosa.f0_harmonics(s, freqs=freqs, f0=f0,
                                       harmonics=np.arange(1, n_harm))

        # normalization
        if self.do_normalize:
            f0, f0_harm = self._normalize(f0, f0_harm)

        features = np.concatenate(([f0], f0_harm), axis=0)

        difference = max_length - features.shape[1]

        # pad or truncate, depending on difference
        if difference > 0:
            features = np.pad(features, ((0, 0), (0, difference)),
                              mode='constant', constant_values=0.)
        elif difference < 0:
            features = features[:, :max_length]

        return np.rot90(features, axes=(1, 0))

    @staticmethod
    def _normalize(f0: np.ndarray, f0_harm: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        pos_mask = f0_harm > 0
        if len(f0_harm[pos_mask]) > 0:
            f0_harm[pos_mask] = np.log10(f0_harm[pos_mask])
            f0_harm[pos_mask] -= np.min(f0_harm[pos_mask])
            f0_harm = f0_harm / np.max(f0_harm)
        else:
            f0_harm = -np.ones(f0_harm.shape)
        f0 = np.nan_to_num(f0, nan=0.)
        if len(f0[f0 > 0]) > 0:
            f0 = f0 / np.max(f0)
        else:
            f0 = -np.ones(f0.shape)
        return f0, f0_harm

    def __call__(
            self,
            raw_speech: Union[np.ndarray, List[float], List[np.ndarray], List[List[float]]],
            sampling_rate: Optional[int] = None,
            return_tensors: Optional[Union[str, TensorType]] = None,
            **kwargs,
    ) -> BatchFeature:
        """
        Main method to featurize and prepare for the model one or several sequence(s).
        Args:
            raw_speech (`np.ndarray`, `List[float]`, `List[np.ndarray]`, `List[List[float]]`):
                The sequence or batch of sequences to be padded. Each sequence can be a numpy array, a list of float
                values, a list of numpy arrays or a list of list of float values.
            sampling_rate (`int`, *optional*):
                The sampling rate at which the `raw_speech` input was sampled. It is strongly recommended to pass
                `sampling_rate` at the forward call to prevent silent errors.
            return_tensors (`str` or [`~utils.TensorType`], *optional*):
                If set, will return tensors instead of list of python integers. Acceptable values are:
                - `'tf'`: Return TensorFlow `tf.constant` objects.
                - `'pt'`: Return PyTorch `torch.Tensor` objects.
                - `'np'`: Return Numpy `np.ndarray` objects.
        """

        if sampling_rate is not None:
            if sampling_rate != self.sampling_rate:
                raise ValueError(
                    f"The model corresponding to this feature extractor: {self} was trained using a sampling rate of"
                    f" {self.sampling_rate}. Please make sure that the provided `raw_speech` input was sampled with"
                    f" {self.sampling_rate} and not {sampling_rate}."
                )
        else:
            pass
            #logger.warning(
            #    "It is strongly recommended to pass the `sampling_rate` argument to this function. "
            #    "Failing to do so can result in silent errors that might be hard to debug."
            #)

        is_batched = bool(
            isinstance(raw_speech, (list, tuple))
            and (isinstance(raw_speech[0], np.ndarray) or isinstance(raw_speech[0], (tuple, list)))
        )

        if is_batched:
            raw_speech = [np.asarray(speech, dtype=np.float32) for speech in raw_speech]
        elif not is_batched and not isinstance(raw_speech, np.ndarray):
            raw_speech = np.asarray(raw_speech, dtype=np.float32)
        elif isinstance(raw_speech, np.ndarray) and raw_speech.dtype is np.dtype(np.float64):
            raw_speech = raw_speech.astype(np.float32)

        # always return batch
        if not is_batched:
            raw_speech = [raw_speech]

        # extract fbank features and pad/truncate to max_length
        features = [self._extract_f0_harmonic_features(waveform, sampling_rate=sampling_rate,
                                                       max_length=self.max_length,
                                                       n_harm=self.n_harmonics) for waveform in raw_speech]

        # convert into BatchFeature
        padded_inputs = BatchFeature({"input_values": features})

        # make sure list is in array format
        input_values = padded_inputs.get("input_values")
        if isinstance(input_values[0], list):
            padded_inputs["input_values"] = [np.asarray(feature, dtype=np.float32) for feature in input_values]

        if return_tensors is not None:
            padded_inputs = padded_inputs.convert_to_tensors(return_tensors)

        return padded_inputs


class Embeddings(nn.Module):
    def __init__(self, input_size, hidden_size, sequence_size) -> None:
        super().__init__()

        self.cls_token = nn.Parameter(
            nn.init.trunc_normal_(torch.zeros(1, 1, hidden_size),
                                  mean=0.0, std=0.02)
        )
        self.embeddings = nn.Sequential(  # nn.Linear(input_size, hidden_size)
            nn.Linear(input_size, hidden_size * 2),
            nn.Linear(hidden_size * 2, hidden_size)
        )
        self.position_embeddings = nn.Parameter(
            nn.init.trunc_normal_(
                torch.zeros(1, sequence_size + 1, hidden_size),
                mean=0.0, std=0.02
            )
        )
        self.dropout = nn.Dropout(0.0)

    def forward(self, inputs):
        # add the [CLS] token to the embedded patch tokens
        cls_tokens = self.cls_token.expand(inputs.shape[0], -1, -1)
        embeddings = torch.cat((cls_tokens, self.embeddings(inputs)), dim=1)  #

        # add positional encoding to each token
        embeddings = embeddings + self.position_embeddings

        # embeddings = self.dropout(embeddings)

        return embeddings


class TransformerNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, sequence_size, num_layers=6):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.embedding = Embeddings(self.input_size, self.hidden_size, sequence_size)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=self.hidden_size,
                                                        nhead=12, batch_first=True)
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=self.num_layers)
        self.classifier = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, x):
        x = self.embedding(x)
        x = self.encoder(x)
        outputs = self.classifier(x[:, 0, :])
        return outputs

