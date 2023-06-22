import numpy as np
import torch
from transformers import AutoFeatureExtractor, ASTForAudioClassification
import librosa


class EmotionRecognition:
    def __init__(self):
        self.feature_extractor = AutoFeatureExtractor.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")
        self.model = ASTForAudioClassification.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593",
                                                               num_labels=4, return_dict=False,
                                                               ignore_mismatched_sizes=True)
        self.model.eval()
        self.to_probs = torch.nn.Softmax(dim=0).requires_grad_(False)

    def recognize(self, waveform, sampling_rate):
        features = self.feature_extractor(waveform, sampling_rate, return_tensors='pt')
        outputs, = self.model(features.input_values.to(torch.float32))
        print(outputs)
        probs = self.to_probs(outputs[0])
        return np.round(probs.detach().numpy() * 100).astype(int)


def check_voiced(waveform, sampling_rate):
    f0, voicing, voicing_p = librosa.pyin(waveform, sr=sampling_rate,
                                          fmin=64., fmax=1046.,
                                          frame_length=512, fill_na=0.)
    print(f0)
    return np.max(f0) > 0
